"""Pure helpers for frozen matrix identity, shared by runner and report gates."""

from __future__ import annotations

from pathlib import Path

from runner.config import ExperimentConfig
from runner.runner import stable_cell_id
from runner.yaml_config import load_yaml_mapping


def expected_cell_ids(config: ExperimentConfig) -> set[str]:
    manifest_path = config.root / config.experiment["dataset_manifest"]
    manifest = load_yaml_mapping(manifest_path, label="dataset selection manifest")
    revision = manifest.get("source", {}).get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("Dataset manifest does not pin a source revision")
    model_alias = config.inference.get("proxy", {}).get("public_model")
    if not isinstance(model_alias, str) or not model_alias:
        models = config.experiment.get("models", [])
        if len(models) != 1:
            raise ValueError("Cannot derive stable IDs without a public model alias")
        model_alias = str(models[0])
    result = set()
    for task_id in config.configured_task_ids:
        for language in config.experiment["languages"]:
            for agent in config.experiment["agents"]:
                for repetition in range(int(config.experiment["repetitions"])):
                    result.add(
                        stable_cell_id(
                            config.experiment_id,
                            revision,
                            str(task_id),
                            str(language),
                            str(agent),
                            model_alias,
                            repetition,
                        )
                    )
    return result


def prepared_task_hashes(task_root: Path) -> dict[str, str]:
    """Hash all pinned task-source bytes, not merely source marker files."""
    import hashlib

    result: dict[str, str] = {}
    for task in sorted(path for path in task_root.iterdir() if path.is_dir()):
        digest = hashlib.sha256()
        count = 0
        for path in sorted(
            (item for item in task.rglob("*") if item.is_file() and "__pycache__" not in item.parts),
            key=lambda item: item.relative_to(task).as_posix(),
        ):
            relative = path.relative_to(task).as_posix().encode("utf-8")
            raw = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
            count += 1
        if count:
            result[task.name] = digest.hexdigest()
    return result
