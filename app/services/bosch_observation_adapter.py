"""Object-specific masked observation contract for Bosch line tests.

These are observations from a test cycle, not a calibrated damage trajectory.
Nominal stage order is metadata; stage durations are unknown.
"""
from __future__ import annotations

import math

FIELD_METADATA = {
    "WP2RunInHalfVolt": ("WP2-half", "V", "condition"),
    "WP2RunInHalfCurrent": ("WP2-half", "A", "response"),
    "WP2RunInFullVolt": ("WP2-full", "V", "condition"),
    "WP2RunInFullCurrent": ("WP2-full", "A", "response"),
    "WP3RunInVolt": ("WP3", "V", "condition"),
    "WP3RunInCurrent": ("WP3", "A", "response"),
    "WP4ImpactEnergy": ("WP4", "m/ss", "proxy_unit_unverified"),
    "WP5ClutchTorq": ("WP5", "Nm", "response"),
    "WP6NoloadVolt": ("WP6", "V", "condition"),
    "WP6NoloadCurrent": ("WP6", "A", "response"),
    "WP7LeakageCurrent1": ("WP7", "mA", "response"),
    "WP7LeakageCurrent2": ("WP7", "mA", "response"),
    "WP1DummyCurrent10": ("WP1", "mA0", "dictionary_unverified"),
    "WP7Similarity": ("WP7", "-", "dictionary_unverified"),
    "WP7Volt": ("WP7", "V", "placeholder_unverified"),
    "WP7Current": ("WP7", "A", "placeholder_unverified"),
}


def build_bosch_observation(values: dict, states: dict | None = None) -> dict:
    """Preserve raw state codes; mask unknown channels without inventing zeros."""
    fields = {}
    for name, (stage, unit, role) in FIELD_METADATA.items():
        value = values.get(name)
        try:
            numeric = float(value)
            valid = math.isfinite(numeric)
        except (TypeError, ValueError):
            numeric, valid = None, False
        verified_role = role in {"condition", "response"}
        fields[name] = {
            "value": numeric if valid else None,
            "available": valid,
            "physical_model_mask": valid and verified_role,
            "stage": stage,
            "unit": unit,
            "role": role,
            "raw_state_code": (states or {}).get(name),
        }
    return {
        "object_type": "bosch_gbh2_24",
        "observation_kind": "factory_test_cycle",
        "stage_durations": None,
        "damage_target": None,
        "fields": fields,
    }
