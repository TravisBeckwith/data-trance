"""
data_trance.context
====================
Every step reads what it needs from prior steps' JSON output if it isn't
already in memory -- this is what makes "run the whole pipeline" and "run
one step standalone" both work off the same config file.
"""

import json
import os

import pandas as pd

from .config import PipelineConfig


class Context:
    def __init__(self, config: PipelineConfig):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "plots"), exist_ok=True)
        self._df = None
        self._cache = {}

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.config.input)
        return self._df

    def path_for(self, step_name: str) -> str:
        return os.path.join(self.config.output_dir, f"{step_name}.json")

    def save(self, step_name: str, data: dict):
        self._cache[step_name] = data
        with open(self.path_for(step_name), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, step_name: str) -> dict:
        if step_name in self._cache:
            return self._cache[step_name]
        path = self.path_for(step_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Step '{step_name}' hasn't been run yet -- expected output at "
                f"{path}. Run it first (e.g. `data-trance step {step_name} "
                f"<config>`) or run the full pipeline."
            )
        with open(path) as f:
            data = json.load(f)
        self._cache[step_name] = data
        return data
