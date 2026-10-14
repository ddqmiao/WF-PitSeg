import math
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


LABELS12_DEFAULT: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14)
HARD_CLASS_RAW_LABELS: Tuple[int, ...] = (3, 9, 12, 14)

# Raw-label boosts. These are defined in raw label space and later mapped to class indices.
CLASS_WEIGHT_BOOSTS_RAW: Dict[int, float] = {
    2: 1.20,
    3: 3.00,
    4: 1.35,
    5: 2.00,
    6: 2.50,
    7: 0.70,
    9: 2.50,
    12: 1.35,
    14: 1.30,
}

# Focus transitions for hard classes and their neighboring chains.
FOCUS_BOUNDARY_RAW_PAIRS: Tuple[Tuple[int, int], ...] = (
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (8, 9),
    (9, 8),
    (9, 10),
    (10, 9),
    (10, 12),
    (12, 10),
    (12, 14),
    (14, 12),
)

DEFAULT_PAIR_TRANSITION_RAW_PAIRS: Tuple[Tuple[int, int], ...] = FOCUS_BOUNDARY_RAW_PAIRS

# Medically curated, directed transition graph. Self loops are handled separately.
ALLOWED_TRANSITIONS_RAW: Dict[int, Set[int]] = {
    1: {2},
    2: {3, 4},
    3: {2, 4},
    4: {2, 3, 5, 6},
    5: {4, 6, 7, 8},
    6: {5, 7, 8},
    7: {5, 6, 8, 9, 10},
    8: {5, 6, 7, 9, 10, 12, 14},
    9: {7, 8, 10, 12},
    10: {8, 9, 12, 14},
    12: {10, 14},
    14: {8, 12},
}


def map_raw_boosts_to_index_boosts(
    labels12: Sequence[int],
    raw_boosts: Mapping[int, float],
) -> Dict[int, float]:
    label_to_index = {int(raw): idx for idx, raw in enumerate(labels12)}
    mapped: Dict[int, float] = {}
    for raw_label, boost in raw_boosts.items():
        class_index = label_to_index.get(int(raw_label))
        if class_index is not None:
            mapped[int(class_index)] = float(boost)
    return mapped


def compress_valid_raw_steps(
    raw_steps: Sequence[int],
    *,
    valid_labels: Optional[Set[int]] = None,
) -> List[int]:
    valid = set(valid_labels) if valid_labels is not None else set(LABELS12_DEFAULT)
    compressed: List[int] = []
    last_value: Optional[int] = None
    for raw_step in raw_steps:
        step = int(raw_step)
        if step not in valid:
            continue
        if last_value != step:
            compressed.append(step)
            last_value = step
    return compressed


def count_segment_transition_counts(
    raw_sequences: Iterable[Sequence[int]],
    *,
    valid_labels: Optional[Set[int]] = None,
    allowed_transitions_raw: Optional[Mapping[int, Set[int]]] = None,
) -> Dict[int, Dict[int, int]]:
    counts: DefaultDict[int, DefaultDict[int, int]] = defaultdict(lambda: defaultdict(int))
    valid = set(valid_labels) if valid_labels is not None else set(LABELS12_DEFAULT)

    for raw_steps in raw_sequences:
        compressed = compress_valid_raw_steps(raw_steps, valid_labels=valid)
        for src_raw, dst_raw in zip(compressed[:-1], compressed[1:]):
            if allowed_transitions_raw is not None:
                if int(dst_raw) not in allowed_transitions_raw.get(int(src_raw), set()):
                    continue
            counts[int(src_raw)][int(dst_raw)] += 1

    return {src: dict(dsts) for src, dsts in counts.items()}


def build_log_transition_matrix_numpy(
    *,
    labels12: Sequence[int],
    transition_counts: Optional[Mapping[int, Mapping[int, int]]] = None,
    allowed_transitions_raw: Optional[Mapping[int, Set[int]]] = None,
    stay_prob: float = 0.98,
    allowed_smoothing: float = 1e-6,
    illegal_log_prob: float = -1e9,
) -> np.ndarray:
    label_list = [int(label) for label in labels12]
    label_to_index = {label: idx for idx, label in enumerate(label_list)}
    num_classes = len(label_list)
    log_trans = np.full((num_classes, num_classes), float(illegal_log_prob), dtype=np.float64)

    if transition_counts is None:
        transition_counts = {}
    if allowed_transitions_raw is None:
        allowed_transitions_raw = {label: set(label_list) - {label} for label in label_list}

    for raw_from in label_list:
        i = label_to_index[raw_from]
        allowed_raw = [
            int(raw_to)
            for raw_to in sorted(allowed_transitions_raw.get(raw_from, set()))
            if int(raw_to) in label_to_index and int(raw_to) != raw_from
        ]
        if not allowed_raw:
            log_trans[i, i] = 0.0
            continue

        row_values: Dict[int, float] = {}
        row_counts = transition_counts.get(raw_from, {})
        total_count = float(sum(int(row_counts.get(raw_to, 0)) for raw_to in allowed_raw))

        if total_count > 0:
            for raw_to in allowed_raw:
                observed = float(row_counts.get(raw_to, 0))
                row_values[raw_to] = max(float(allowed_smoothing), observed / total_count * (1.0 - stay_prob))
        else:
            uniform_mass = (1.0 - stay_prob) / max(1, len(allowed_raw))
            for raw_to in allowed_raw:
                row_values[raw_to] = max(float(allowed_smoothing), uniform_mass)

        row_sum = sum(row_values.values())
        if row_sum <= 0:
            uniform_mass = (1.0 - stay_prob) / max(1, len(allowed_raw))
            for raw_to in allowed_raw:
                row_values[raw_to] = uniform_mass
            row_sum = sum(row_values.values())

        scale = (1.0 - stay_prob) / row_sum
        for raw_to in allowed_raw:
            j = label_to_index[raw_to]
            prob = max(float(allowed_smoothing), row_values[raw_to] * scale)
            log_trans[i, j] = math.log(prob)
        log_trans[i, i] = math.log(float(stay_prob))

    return log_trans
