from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.yaml_config import load_yaml_mapping


@dataclass(slots=True)
class ExperimentConfig:
    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent.parent.resolve()

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        resolved = path.resolve()
        data = load_yaml_mapping(resolved, label="experiment configuration")
        for section in ("experiment", "generation", "sandbox", "storage"):
            if not isinstance(data.get(section), dict):
                raise ValueError(f"Experiment configuration requires a {section} mapping")
        experiment = data["experiment"]
        for key in (
            "id", "dataset_manifest", "prepared_tasks_root", "tasks", "languages",
            "agents", "models", "repetitions", "seed",
        ):
            if key not in experiment:
                raise ValueError(f"Experiment configuration is missing experiment.{key}")
        if not isinstance(experiment["tasks"], list) or not experiment["tasks"]:
            raise ValueError("experiment.tasks must be a non-empty list")
        if len(experiment["tasks"]) != len(set(experiment["tasks"])):
            raise ValueError("experiment.tasks must contain unique task IDs")
        return cls(resolved, data)

    @property
    def experiment(self) -> dict[str, Any]:
        return self.data["experiment"]

    @property
    def is_corrected_phase1(self) -> bool:
        return str(self.experiment.get("version", "")).startswith("corrected-v")

    @property
    def generation(self) -> dict[str, Any]:
        return self.data["generation"]

    @property
    def sandbox(self) -> dict[str, Any]:
        return self.data["sandbox"]

    @property
    def storage(self) -> dict[str, Any]:
        return self.data["storage"]

    @property
    def inference(self) -> dict[str, Any]:
        return self.data.get("inference", {})

    @property
    def task_root(self) -> Path:
        configured = self.experiment.get("prepared_tasks_root", "benchmark/tasks")
        return (self.root / configured).resolve()

    @property
    def experiment_id(self) -> str:
        return str(self.experiment.get("id", self.experiment.get("name", self.path.stem)))

    @property
    def agents_manifest(self) -> Path:
        return (self.root / self.experiment.get("agents_manifest", "configs/agents.yaml")).resolve()

    @property
    def models_manifest(self) -> Path:
        return (self.root / self.experiment.get("models_manifest", "configs/models.yaml")).resolve()

    @property
    def configured_task_ids(self) -> list[str]:
        values = self.experiment.get("tasks", [])
        if not isinstance(values, list) or not values:
            raise ValueError("experiment.tasks must be a non-empty list")
        return [str(value) for value in values]
