from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent.parent.resolve()

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        with path.open("r", encoding="utf-8") as handle:
            return cls(path.resolve(), yaml.safe_load(handle))

    @property
    def experiment(self) -> dict[str, Any]:
        return self.data["experiment"]

    @property
    def generation(self) -> dict[str, Any]:
        return self.data["generation"]

    @property
    def sandbox(self) -> dict[str, Any]:
        return self.data["sandbox"]

    @property
    def storage(self) -> dict[str, Any]:
        return self.data["storage"]
