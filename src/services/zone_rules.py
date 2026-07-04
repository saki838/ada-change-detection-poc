"""Zone rules engine — severity scoring and compliance rules per zone type.

The UPLC scope doc for Agra Development Authority identifies several zone
types with different enforcement priorities:
  - Heritage zones: maximum scrutiny, critical severity boost
  - Green belt / Riverfront: high priority, special setback rules
  - Residential / Commercial: standard enforcement
  - Industrial: standard with area-based modifiers

Severity scoring considers:
  1. Zone type base severity boost
  2. Violation class severity weight
  3. Area/size multiplier
  4. Repeat-offender flag (future)

Usage:
    from src.services.zone_rules import ZoneRulesEngine

    rules = ZoneRulesEngine()
    severity = rules.compute_severity(
        zone_type="heritage",
        violation_class="new_construction",
        area_m2=150.0,
    )
    # -> "critical"
"""
from __future__ import annotations

from typing import Optional


# ── Default zone configurations ─────────────────────────────────────
# These map to the UPLC scope document's zone classifications.
# In Phase 2+, these will also be loaded dynamically from zone_boundaries
# stored in the database.

ZONE_CONFIGS = {
    "heritage": {
        "label": "Heritage Zone",
        "base_severity": "high",
        "severity_boost": 2.0,
        "setback_distance_m": 6.0,
        "description": "Heritage-sensitive areas requiring maximum scrutiny (Section 4.2)",
    },
    "green_belt": {
        "label": "Green Belt",
        "base_severity": "medium",
        "severity_boost": 1.5,
        "setback_distance_m": 5.0,
        "description": "Environmental/green-buffer zones (Yamuna riverfront, parks)",
    },
    "riverfront": {
        "label": "Riverfront Buffer",
        "base_severity": "high",
        "severity_boost": 1.8,
        "setback_distance_m": 10.0,
        "description": "Yamuna riverfront buffer zone — special setback rules apply",
    },
    "residential": {
        "label": "Residential Zone",
        "base_severity": "low",
        "severity_boost": 1.0,
        "setback_distance_m": 3.0,
        "description": "Standard residential areas",
    },
    "commercial": {
        "label": "Commercial Zone",
        "base_severity": "medium",
        "severity_boost": 1.2,
        "setback_distance_m": 4.0,
        "description": "Commercial / mixed-use areas",
    },
    "industrial": {
        "label": "Industrial Zone",
        "base_severity": "low",
        "severity_boost": 1.0,
        "setback_distance_m": 5.0,
        "description": "Industrial / manufacturing areas",
    },
    "other": {
        "label": "Unclassified Zone",
        "base_severity": "medium",
        "severity_boost": 1.0,
        "setback_distance_m": 3.0,
        "description": "Unclassified or mixed zones — standard rules apply",
    },
}

# Violation class severity weights (higher = more severe)
VIOLATION_WEIGHTS = {
    "new_construction": 1.5,
    "horizontal_expansion": 1.3,
    "vertical_expansion": 1.4,
    "encroachment": 1.6,
    "unauthorized_paving": 1.1,
    "vegetation_clearance": 1.2,
    "other_change": 1.0,
}

# Area thresholds for scaling severity (in m²)
AREA_THRESHOLDS = [
    (50, 0.0),     # < 50 m²: no boost
    (200, 0.5),    # 50–200 m²: +0.5
    (500, 1.0),    # 200–500 m²: +1.0
    (float("inf"), 1.5),  # > 500 m²: +1.5
]

SEVERITY_ORDER = ["low", "medium", "high", "critical"]


def _severity_index(severity: str) -> int:
    """Convert severity string to numeric index for comparison."""
    try:
        return SEVERITY_ORDER.index(severity.lower())
    except ValueError:
        return 0


def _index_to_severity(index: int) -> str:
    """Clamp an index to valid severity range and return the label."""
    return SEVERITY_ORDER[max(0, min(index, len(SEVERITY_ORDER) - 1))]


class ZoneRulesEngine:
    """Computes severity and checks compliance rules per zone type.

    This is the pure-logic engine (no database dependency). Zone configs
    can be overridden at construction time to support DB-driven config in
    later phases.
    """

    def __init__(self, zone_configs: Optional[dict] = None):
        self._zone_configs = zone_configs or ZONE_CONFIGS

    # ── Public API ──────────────────────────────────────────────

    def compute_severity(
        self,
        zone_type: str,
        violation_class: str,
        area_m2: float,
    ) -> str:
        """Compute the overall severity of a violation.

        Combines:
          - Zone type base severity + boost
          - Violation class weight
          - Area/size modifier

        Returns one of: low, medium, high, critical
        """
        zone = self._zone_configs.get(zone_type, self._zone_configs["other"])
        base_idx = _severity_index(zone["base_severity"])

        # Apply zone severity boost (as a multiplicative index shift)
        zone_shift = int(round((zone["severity_boost"] - 1.0) * 2))

        # Apply violation class weight
        vw = VIOLATION_WEIGHTS.get(violation_class, 1.0)
        class_shift = int(round((vw - 1.0) * 2))

        # Apply area multiplier
        area_shift = 0
        for threshold, boost in AREA_THRESHOLDS:
            if area_m2 <= threshold:
                area_shift = int(round(boost))
                break

        total_idx = base_idx + zone_shift + class_shift + area_shift
        return _index_to_severity(total_idx)

    def get_setback_distance(self, zone_type: str) -> float:
        """Return the required setback distance in meters for a zone type."""
        zone = self._zone_configs.get(zone_type, self._zone_configs["other"])
        return zone["setback_distance_m"]

    def get_zone_boost(self, zone_type: str) -> float:
        """Return the severity boost multiplier for a zone type."""
        zone = self._zone_configs.get(zone_type, self._zone_configs["other"])
        return zone["severity_boost"]

    def get_zone_label(self, zone_type: str) -> str:
        """Return a human-readable label for a zone type."""
        zone = self._zone_configs.get(zone_type, self._zone_configs["other"])
        return zone["label"]

    def list_zone_types(self) -> list[str]:
        """Return all configured zone type keys."""
        return list(self._zone_configs.keys())

    def list_active_zones(self) -> list[dict]:
        """Return all zone configs as a list of dicts (for API responses)."""
        return [
            {
                "zone_type": k,
                "label": v["label"],
                "base_severity": v["base_severity"],
                "severity_boost": v["severity_boost"],
                "setback_distance_m": v["setback_distance_m"],
                "description": v["description"],
            }
            for k, v in self._zone_configs.items()
        ]

    # ── Compliance checks ───────────────────────────────────────

    def check_setback_compliance(
        self,
        zone_type: str,
        measured_setback_m: Optional[float],
    ) -> dict:
        """Check if a measured setback distance complies with zone rules.

        Returns:
            {
                "compliant": bool,
                "required_m": float,
                "measured_m": float | None,
                "violation_m": float | None,  # negative if compliant
            }
        """
        required = self.get_setback_distance(zone_type)
        if measured_setback_m is None:
            return {
                "compliant": False,
                "required_m": required,
                "measured_m": None,
                "violation_m": None,
            }
        return {
            "compliant": measured_setback_m >= required,
            "required_m": required,
            "measured_m": round(measured_setback_m, 2),
            "violation_m": round(required - measured_setback_m, 2)
            if measured_setback_m < required
            else None,
        }

    def check_zone_restriction(
        self,
        zone_type: str,
        violation_class: str,
    ) -> dict:
        """Check if a violation class is restricted in a given zone.

        For example, any new construction in a heritage zone is an automatic
        critical violation, whereas vegetation clearance in a green belt
        might have special rules.

        Returns:
            {
                "restricted": bool,
                "severity_override": str | None,  # force severity if restricted
                "reason": str | None,
            }
        """
        # Heritage: any construction-related violation is automatically critical
        if zone_type == "heritage" and violation_class in (
            "new_construction",
            "horizontal_expansion",
            "vertical_expansion",
        ):
            return {
                "restricted": True,
                "severity_override": "critical",
                "reason": "New construction or expansion detected in a heritage-sensitive zone.",
            }

        # Green belt / riverfront: new construction restricted
        if zone_type in ("green_belt", "riverfront") and violation_class == "new_construction":
            return {
                "restricted": True,
                "severity_override": "critical",
                "reason": f"New construction in {self.get_zone_label(zone_type)} is prohibited.",
            }

        return {
            "restricted": False,
            "severity_override": None,
            "reason": None,
        }
