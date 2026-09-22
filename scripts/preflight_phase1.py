from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.loader import select_tasks
from runner.cli import load_runtime
from runner.config import ExperimentConfig
from runner.preflight import full_preflight
from scripts.prepare_phase1 import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Phase I preflight gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Pinned Harness-Bench checkout")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    result: dict[str, object] = {}
    if args.source:
        result["dataset"] = prepare(
            args.source.resolve(),
            config.root / "benchmark/task_selection.yaml",
            config.root / "benchmark/translations/phase1.yaml",
            config.task_root,
            check_only=True,
        )
    agents, models, _ = load_runtime(config)
    tasks = select_tasks(config.task_root, config.configured_task_ids)
    result.update(full_preflight(config, tasks, agents, models))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
