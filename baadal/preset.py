"""Load a region preset (terrain, scenarios, zones, intervention catalog) from JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PRESET_DIR = Path(__file__).resolve().parent.parent / "presets"


@dataclass
class Intervention:
    id: str
    name: str
    category: str
    cost_pkr_m: float
    location: tuple[int, int]
    damage_reduction: dict[str, float]
    people_reduction: dict[str, float]
    description: str = ""
    # Damage / exposure an action CREATES elsewhere (e.g. a controlled breach floods dry farmland),
    # as a fraction of the zone's total asset value / population.
    damage_added: dict[str, float] = field(default_factory=dict)
    people_added: dict[str, float] = field(default_factory=dict)


@dataclass
class Preset:
    raw: dict
    id: str
    name: str
    interventions: list[Intervention]
    zones: list[dict]
    scenarios: list[dict]
    synergies: list[dict] = field(default_factory=list)
    exclusions: list[dict] = field(default_factory=list)
    dependencies: list[dict] = field(default_factory=list)

    @property
    def zone_ids(self) -> list[str]:
        return [z["id"] for z in self.zones]

    @property
    def intervention_ids(self) -> list[str]:
        return [iv.id for iv in self.interventions]

    def index(self, intervention_id: str) -> int:
        return self.intervention_ids.index(intervention_id)

    def scenario(self, scenario_id: str | None = None) -> dict:
        sid = scenario_id or self.raw["default_scenario"]
        for s in self.scenarios:
            if s["id"] == sid:
                return s
        raise KeyError(f"Unknown scenario {sid!r}")

    @property
    def cost_unit(self) -> float:
        return float(self.raw["cost_unit_pkr_m"])

    @property
    def people_weight(self) -> float:
        return float(self.raw["people_weight_pkr_m_per_person"])

    @property
    def budgets(self) -> list[float]:
        return list(self.raw["budgets_pkr_m"])


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def load_preset(preset_id: str = "riverine_punjab") -> Preset:
    path = PRESET_DIR / f"{preset_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    ivs = [
        Intervention(
            id=d["id"],
            name=d["name"],
            category=d["category"],
            cost_pkr_m=float(d["cost_pkr_m"]),
            location=tuple(d["location"]),
            damage_reduction=dict(d.get("damage_reduction", {})),
            people_reduction=dict(d.get("people_reduction", {})),
            description=d.get("description", ""),
            damage_added=dict(d.get("damage_added_frac_of_assets", {})),
            people_added=dict(d.get("people_added_frac_of_population", {})),
        )
        for d in raw["interventions"]
    ]
    preset = Preset(
        raw=raw,
        id=raw["id"],
        name=raw["name"],
        interventions=ivs,
        zones=raw["zones"],
        scenarios=raw["scenarios"],
        synergies=raw.get("synergies", []),
        exclusions=raw.get("exclusions", []),
        dependencies=raw.get("dependencies", []),
    )
    _validate(preset)
    return preset


def _validate(p: Preset) -> None:
    zids = set(p.zone_ids)
    ids = set(p.intervention_ids)
    unit = p.cost_unit
    for iv in p.interventions:
        bad = (set(iv.damage_reduction) | set(iv.people_reduction) | set(iv.damage_added) | set(iv.people_added)) - zids
        if bad:
            raise ValueError(f"{iv.id} references unknown zones {bad}")
        if abs(iv.cost_pkr_m / unit - round(iv.cost_pkr_m / unit)) > 1e-9:
            raise ValueError(f"{iv.id} cost {iv.cost_pkr_m} is not a multiple of the cost unit {unit}")
        for r in list(iv.damage_reduction.values()) + list(iv.people_reduction.values()):
            if not -1.0 <= r < 1.0:
                raise ValueError(f"{iv.id} has reduction {r} outside [-1, 1)")
    for rule in p.synergies + p.exclusions:
        if not set(rule["pair"]) <= ids:
            raise ValueError(f"Rule references unknown interventions: {rule}")
    for rule in p.dependencies:
        if not {rule["if"], rule["requires"]} <= ids:
            raise ValueError(f"Dependency references unknown interventions: {rule}")
