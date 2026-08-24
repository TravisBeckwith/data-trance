"""
data_trance.config
===================
Loads and validates the pipeline's YAML config file.
"""

from dataclasses import dataclass, field

import yaml

VALID_TYPES = {"auto", "continuous", "count", "proportion", "correlation",
                "categorical", "ordinal"}
VALID_STEPS = ["detect_type", "assess", "recommend", "apply_transform",
               "validate", "report"]


@dataclass
class ColumnConfig:
    name: str
    type: str = "auto"
    transform: str | None = None  # override the recommended transform


@dataclass
class PipelineConfig:
    input: str
    output_dir: str = "results"
    alpha: float = 0.05
    recommend_method: str = "rule_based"   # rule_based | ml | dl
    ml_min_confidence: float = 0.5
    columns: dict = field(default_factory=dict)   # name -> ColumnConfig
    steps: list = field(default_factory=lambda: list(VALID_STEPS))
    raw: dict = field(default_factory=dict)

    def column_type(self, name: str) -> str:
        col = self.columns.get(name)
        return col.type if col else "auto"

    def column_transform_override(self, name: str):
        col = self.columns.get(name)
        return col.transform if col else None


def load_config(path: str) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if "input" not in raw:
        raise ValueError(f"Config at {path} is missing required key 'input'")

    columns = {}
    for name, spec in (raw.get("columns") or {}).items():
        spec = spec or {}
        ctype = spec.get("type", "auto")
        if ctype not in VALID_TYPES:
            raise ValueError(
                f"Column '{name}': invalid type '{ctype}'. "
                f"Must be one of {sorted(VALID_TYPES)}"
            )
        columns[name] = ColumnConfig(name=name, type=ctype,
                                      transform=spec.get("transform"))

    steps = raw.get("steps", list(VALID_STEPS))
    for s in steps:
        if s not in VALID_STEPS:
            raise ValueError(f"Unknown step '{s}'. Must be one of {VALID_STEPS}")

    recommend_method = raw.get("recommend_method", "rule_based")
    if recommend_method not in ("rule_based", "ml", "dl"):
        raise ValueError(
            f"Invalid recommend_method '{recommend_method}'. "
            f"Must be one of: rule_based, ml, dl"
        )

    return PipelineConfig(
        input=raw["input"],
        output_dir=raw.get("output_dir", "results"),
        alpha=float(raw.get("alpha", 0.05)),
        recommend_method=recommend_method,
        ml_min_confidence=float(raw.get("ml_min_confidence", 0.5)),
        columns=columns,
        steps=steps,
        raw=raw,
    )
