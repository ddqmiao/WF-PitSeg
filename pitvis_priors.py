from typing import Dict, List, Mapping, Optional


# Locked emission multipliers used by the offline fixed-prior evaluation.
# They are historical development choices and are not re-tuned per CV fold.
DEFAULT_EMISSION_PRIOR_BY_RAW_LABEL: Dict[int, float] = {
    3: 8.0,
    5: 3.0,
    6: 2.2,
    7: 0.20,
    8: 0.90,
    9: 5.50,
    14: 1.50,
}

DEFAULT_ENSEMBLE_PRESET_NAME = "single_model_default"

EMISSION_PRIOR_PRESETS_BY_RAW_LABEL: Dict[str, Dict[int, float]] = {
    DEFAULT_ENSEMBLE_PRESET_NAME: DEFAULT_EMISSION_PRIOR_BY_RAW_LABEL,
}

ENSEMBLE_WEIGHT_PRESETS: Dict[str, List[float]] = {
    DEFAULT_ENSEMBLE_PRESET_NAME: [1.0],
}


def build_emission_prior_by_index(
    label_to_index: Mapping[int, int],
    preset: Optional[str] = None,
) -> Dict[int, float]:
    """Convert raw-label emission multipliers to class-index keys."""
    raw_prior = (
        DEFAULT_EMISSION_PRIOR_BY_RAW_LABEL
        if preset is None
        else EMISSION_PRIOR_PRESETS_BY_RAW_LABEL[preset]
    )
    return {
        int(label_to_index[int(raw_label)]): float(multiplier)
        for raw_label, multiplier in raw_prior.items()
    }


def build_ensemble_weight_preset(preset: str) -> List[float]:
    """Return checkpoint weights for a named compatibility preset."""
    return [float(weight) for weight in ENSEMBLE_WEIGHT_PRESETS[preset]]


__all__ = [
    "DEFAULT_EMISSION_PRIOR_BY_RAW_LABEL",
    "DEFAULT_ENSEMBLE_PRESET_NAME",
    "EMISSION_PRIOR_PRESETS_BY_RAW_LABEL",
    "ENSEMBLE_WEIGHT_PRESETS",
    "build_emission_prior_by_index",
    "build_ensemble_weight_preset",
]
