from typing import Dict, Optional, Sequence, Tuple


COARSE_STAGE_NAMES: Tuple[str, ...] = ("early", "middle", "late")
COARSE_STAGE_LABEL_GROUPS: Tuple[Tuple[int, ...], ...] = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 12, 14),
)


def build_raw_label_to_coarse_index(
    groups: Sequence[Sequence[int]] = COARSE_STAGE_LABEL_GROUPS,
) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for coarse_idx, labels in enumerate(groups):
        for raw_label in labels:
            mapping[int(raw_label)] = int(coarse_idx)
    return mapping


def build_fine_to_coarse_index(
    fine_labels: Sequence[int],
    *,
    raw_label_to_coarse_index: Optional[Dict[int, int]] = None,
) -> Tuple[int, ...]:
    mapping = raw_label_to_coarse_index or build_raw_label_to_coarse_index()
    return tuple(int(mapping[int(raw_label)]) for raw_label in fine_labels)


__all__ = [
    "COARSE_STAGE_LABEL_GROUPS",
    "COARSE_STAGE_NAMES",
    "build_fine_to_coarse_index",
    "build_raw_label_to_coarse_index",
]
