from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from analysis.corrected_report import build_corrected_report
from analysis.outcome_pdf_v2 import build_outcome_pdf_v2
from analysis.outcome_pdf_v8 import build_v8_pdf
from analysis.outcome_report_v2 import build_outcome_report_v2
from analysis.outcome_report_v8 import build_v8_tables
from analysis.report import build_phase1_report
from benchmark.loader import discover_tasks, select_tasks
from runner.config import ExperimentConfig
from runner.inference import ensure_model_manifest, load_env_file
from runner.judge import judge_database
from runner.outcome_calibration import build_calibration_cases
from runner.outcome_calibration_finalize import finalize_calibration_store
from runner.outcome_judge import sha256_text
from runner.outcome_v2 import (
    OutcomeJudgeRuntime,
    _check_oracle_controls,
    _grade_control_oracle,
    _scan_for_secrets,
    preflight_saved_pilot,
    run_calibration_v2,
    run_judgments_v2,
    verify_frozen_inputs,
)
from runner.outcome_v8 import calibrate as calibrate_outcomes_v8
from runner.outcome_v8 import rejudge as rejudge_outcomes_v8
from runner.preflight import PreflightError, check_pilot_fixture_parity, full_preflight
from runner.redaction import redact_text
from runner.runner import ExperimentRunner

app = typer.Typer(no_args_is_help=True)
console = Console()
OUTCOME_DIR = Path("data/phase1/corrected/pilot-v14/rejudgments/outcome-v7")
OUTCOME_RUBRIC = Path("configs/outcome-rubric.pilot-v14-v7.yaml")
OUTCOME_V8_DIR = Path("data/phase1/corrected/pilot-v14/rejudgments/outcome-v8")
OUTCOME_V8_RUBRIC = Path("configs/outcome-rubric.pilot-v14-v8.yaml")


def _safe_command_error(label: str, exc: Exception, model_config: dict | None = None) -> None:
    secrets = ()
    if model_config:
        secrets = tuple(
            value
            for value in (
                model_config.get("api_key"),
                model_config.get("base_url"),
                model_config.get("model"),
            )
            if isinstance(value, str) and value
        )
    message = redact_text(str(exc), secrets) if model_config is not None else "details withheld until credentials are loaded"
    console.print(f"{label} failed: {type(exc).__name__}: {message}")


def _runtime_secret_mapping(runtime: OutcomeJudgeRuntime | None) -> dict | None:
    if runtime is None or len(runtime.secrets) < 3:
        return None
    return {
        "base_url": runtime.secrets[0],
        "api_key": runtime.secrets[1],
        "model": runtime.secrets[2],
    }


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_runtime(config: ExperimentConfig) -> tuple[dict, dict, dict | None]:
    root = config.root
    agents = load_yaml(config.agents_manifest)["agents"]
    models = load_yaml(config.models_manifest)["models"]
    model_manifest = None
    for model_name in config.experiment.get("models", []):
        model_config = models.get(model_name, {})
        if model_config.get("provider") == "university_gpu" or model_name == "university_gpu":
            endpoint, model_manifest = ensure_model_manifest(
                root, config.inference, verify_tool=True
            )
            models[model_name] = endpoint.model_config()
    return agents, models, model_manifest


@app.command("list-tasks")
def list_tasks(tasks_dir: Path = typer.Option(Path("benchmark/tasks"), exists=True)) -> None:
    table = Table("Task", "Category", "Source")
    for task in discover_tasks(tasks_dir):
        table.add_row(task.task_id, task.category, task.source_task or "")
    console.print(table)


@app.command("run")
def run(
    config: Path = typer.Option(Path("configs/phase1.research.yaml"), exists=True),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    max_cells: int | None = typer.Option(None, min=1),
) -> None:
    experiment_config = ExperimentConfig.load(config)
    agents, models, _ = load_runtime(experiment_config)
    task_ids = experiment_config.configured_task_ids
    task_defs = select_tasks(experiment_config.task_root, task_ids)
    try:
        gate = full_preflight(experiment_config, task_defs, agents, models)
    except PreflightError as exc:
        console.print(f"Preflight failed: {exc}")
        raise typer.Exit(code=2) from exc
    console.print(
        f"Preflight passed: {gate['matrix']['cells']} cells, "
        f"{gate['prepared_tasks']} prepared tasks and {gate['language_variants']} variants."
    )
    runner = ExperimentRunner(experiment_config, agents, models)
    try:
        matrix = runner.validate_matrix(task_defs)
        results = runner.run(task_defs, resume=resume, max_cells=max_cells)
        summary = runner.completion_summary(task_defs)
    finally:
        runner.close()
    successful = sum(1 for result in results if result["status"] == "completed" and result["success"])
    completed = sum(1 for result in results if result["status"] == "completed")
    console.print(
        f"Matrix {matrix['cells']} cells; processed {len(results)}; "
        f"completed {completed}, perfect-score {successful}, "
        f"infrastructure errors {summary['infrastructure_error']}, pending {summary['pending']}."
    )
    if max_cells is None and (
        summary["gradable"] != summary["expected"] or summary["infrastructure_error"]
    ):
        raise typer.Exit(code=2)


@app.command("report")
def report(
    database: Path = typer.Option(Path("data/phase1/experiment/runs.sqlite"), exists=True),
    output: Path = typer.Option(Path("data/phase1/experiment/provisional_report.md")),
) -> None:
    build_phase1_report(database, output)
    console.print(f"Wrote {output}")


@app.command("corrected-report")
def corrected_report(
    config: Path = typer.Option(Path("configs/phase1.corrected.research.yaml"), exists=True),
    database: Path | None = typer.Option(None),
    output: Path = typer.Option(Path("data/phase1/corrected/provisional_report.md")),
    experiment_id: str | None = typer.Option(None),
) -> None:
    experiment_config = ExperimentConfig.load(config)
    db = database or (experiment_config.root / experiment_config.storage["database"])
    pristine = None
    if experiment_config.is_corrected_phase1:
        tasks = select_tasks(experiment_config.task_root, experiment_config.configured_task_ids)
        pristine = check_pilot_fixture_parity(experiment_config, tasks)["pristine_oracle_scores"]
    result = build_corrected_report(db, output, experiment_id or experiment_config.experiment_id,
                                    pristine_oracle_scores=pristine)
    console.print(
        f"Wrote {output}; {result['cells']['completed_rows']} completed cells, "
        f"{result['cells']['oracle_gradable']} oracle-gradable."
    )


@app.command("judge")
def judge(
    config: Path = typer.Option(Path("configs/phase1.corrected.research.yaml"), exists=True),
    database: Path | None = typer.Option(None),
    experiment_id: str | None = typer.Option(None),
    rerun: bool = typer.Option(False, "--rerun", help="Recompute all process judgments under the current frozen normalizer."),
) -> None:
    """Run the frozen paper-style process rubric after agent execution."""
    experiment_config = ExperimentConfig.load(config)
    agents, models, _ = load_runtime(experiment_config)
    runner = ExperimentRunner(experiment_config, agents, models)
    try:
        runner.assert_experiment_identity()
        model_name = str(experiment_config.experiment["models"][0])
        result = judge_database(
            experiment_config.root / (database or experiment_config.storage["database"]),
            experiment_config.task_root,
            runner.models[model_name],
            experiment_id=experiment_id or experiment_config.experiment_id,
            max_tokens=int(experiment_config.generation.get("max_tokens", 2048)),
            proxy=runner.proxies.get(model_name),
            rerun=rerun,
        )
    finally:
        runner.close()
    console.print(result)


@app.command("judge-outcomes")
def judge_outcomes(
    config: Path = typer.Option(Path("configs/phase1.corrected.pilot-v14.yaml"), exists=True),
    source_database: Path = typer.Option(Path("data/phase1/corrected/pilot-v14/runs.sqlite"), exists=True),
    rubric: Path = typer.Option(OUTCOME_RUBRIC, exists=True),
    calibration: Path = typer.Option(OUTCOME_DIR / "calibration.json", exists=True),
    store: Path = typer.Option(OUTCOME_DIR / "judgments.sqlite"),
    experiment_id: str = typer.Option("phase1_corrected_pilot_v14"),
    seed: int = typer.Option(1701),
    max_cells: int | None = typer.Option(None, min=1),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Score saved final workspaces with the frozen LLM outcome rubric."""
    runtime = None
    try:
        experiment_config = ExperimentConfig.load(config)
        source = experiment_config.root / source_database
        verify_frozen_inputs(experiment_config.root, source)
        runtime = OutcomeJudgeRuntime(experiment_config.root, experiment_config.inference)
        with runtime:
            result = run_judgments_v2(
                root=experiment_config.root,
                source_database=source,
                experiment_id=experiment_id,
                experiment_config=experiment_config.data,
                task_root=experiment_config.task_root,
                workspace_image=str(experiment_config.sandbox["image"]),
                rubric_path=experiment_config.root / rubric,
                calibration_path=experiment_config.root / calibration,
                store_path=experiment_config.root / store,
                runtime=runtime,
                seed=seed,
                resume=resume,
                max_cells=max_cells,
            )
    except Exception as exc:
        message = redact_text(str(exc), runtime.secrets) if runtime and runtime.secrets else "Details withheld until the private inference settings are loaded."
        console.print(f"Outcome judging failed: {type(exc).__name__}: {message}")
        raise typer.Exit(code=2) from None
    console.print(result)


@app.command("calibrate-outcomes")
def calibrate_outcomes(
    config: Path = typer.Option(Path("configs/phase1.corrected.pilot-v14.yaml"), exists=True),
    rubric: Path = typer.Option(OUTCOME_RUBRIC, exists=True),
    store: Path = typer.Option(OUTCOME_DIR / "calibration.sqlite"),
    output: Path = typer.Option(OUTCOME_DIR / "calibration.json"),
    source_database: Path = typer.Option(Path("data/phase1/corrected/pilot-v14/runs.sqlite"), exists=True),
    experiment_id: str = typer.Option("phase1_corrected_pilot_v14"),
    seed: int = typer.Option(1701),
) -> None:
    """Run isolated deterministic and LLM controls before judging pilot workspaces."""
    runtime = None
    try:
        experiment_config = ExperimentConfig.load(config)
        source = experiment_config.root / source_database
        verify_frozen_inputs(experiment_config.root, source)
        runtime = OutcomeJudgeRuntime(experiment_config.root, experiment_config.inference)
        with runtime:
            report = run_calibration_v2(
                root=experiment_config.root,
                source_database=source,
                experiment_id=experiment_id,
                experiment_config=experiment_config.data,
                task_root=experiment_config.task_root,
                workspace_image=str(experiment_config.sandbox["image"]),
                rubric_path=experiment_config.root / rubric,
                store_path=experiment_config.root / store,
                output_path=experiment_config.root / output,
                runtime=runtime,
                seed=seed,
            )
    except Exception as exc:
        message = redact_text(str(exc), runtime.secrets) if runtime and runtime.secrets else "Details withheld until the private inference settings are loaded."
        console.print(f"Outcome calibration failed: {type(exc).__name__}: {message}")
        raise typer.Exit(code=2) from None
    console.print({"status": report["status"], "control_count": report["control_count"], "checks": report["checks"]})


@app.command("finalize-outcome-calibration")
def finalize_outcome_calibration(
    config: Path = typer.Option(Path("configs/phase1.corrected.pilot-v14.yaml"), exists=True),
    rubric: Path = typer.Option(OUTCOME_RUBRIC, exists=True),
    store: Path = typer.Option(OUTCOME_DIR / "calibration.sqlite", exists=True),
    output: Path = typer.Option(OUTCOME_DIR / "calibration.json"),
    source_database: Path = typer.Option(
        Path("data/phase1/corrected/pilot-v14/runs.sqlite"), exists=True
    ),
    seed: int = typer.Option(1701),
) -> None:
    """Finalize all completed controls without making another model call."""
    experiment_config = ExperimentConfig.load(config)
    root = experiment_config.root
    inference = experiment_config.inference
    try:
        private = load_env_file(root / inference.get("env_file", ".env.uni-gpu.local"))
        pinned = json.loads((root / inference["manifest"]).read_text(encoding="utf-8"))
        resolved_model = pinned["resolved_model"]
        secrets = tuple(
            value
            for value in (
                private.get("INDIC_UNI_GPU_BASE_URL"),
                private.get("INDIC_UNI_GPU_API_KEY"),
                resolved_model,
            )
            if isinstance(value, str) and value
        )
        report = finalize_calibration_store(
            root=root,
            source_database=root / source_database,
            rubric_path=root / rubric,
            task_root=experiment_config.task_root,
            store_path=root / store,
            output_path=root / output,
            public_model_alias=str((inference.get("proxy") or {}).get("public_model")),
            model_identity_sha256=sha256_text(resolved_model),
            secrets=secrets,
            seed=seed,
        )
    except Exception as exc:
        console.print(f"Calibration finalization failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=2) from None
    console.print(
        {
            "status": report["status"],
            "control_count": report["control_count"],
            "checks_passed": sum(item["passed"] for item in report["checks"]),
            "checks_failed": sum(not item["passed"] for item in report["checks"]),
            "finalized_from_append_only_store": report["finalized_from_append_only_store"],
        }
    )


@app.command("outcome-preflight")
def outcome_preflight(
    config: Path = typer.Option(Path("configs/phase1.corrected.pilot-v14.yaml"), exists=True),
    source_database: Path = typer.Option(Path("data/phase1/corrected/pilot-v14/runs.sqlite"), exists=True),
    experiment_id: str = typer.Option("phase1_corrected_pilot_v14"),
) -> None:
    """Validate frozen sources, every saved archive/trace, Docker tests, and deterministic controls."""
    try:
        experiment_config = ExperimentConfig.load(config)
        source = experiment_config.root / source_database
        verify_frozen_inputs(experiment_config.root, source)
        cells, packets, hashes = preflight_saved_pilot(
            experiment_config.root,
            source,
            experiment_id,
            experiment_config.task_root,
            experiment_config.data,
            str(experiment_config.sandbox["image"]),
        )
        cases = build_calibration_cases(experiment_config.task_root)
        oracle_scores = {
            case["case_id"]: _grade_control_oracle(
                case["task_id"], case["files"], experiment_config.task_root,
                str(experiment_config.sandbox["image"]),
            )
            for case in cases
        }
        oracle_checks = _check_oracle_controls(oracle_scores, cases)
        failed = [check for check in oracle_checks if not check["passed"]]
        summary = {
            "source_cells": len(cells),
            "archived_packets": len(packets),
            "source_trace_hashes": sum(key.startswith("trace:") for key in hashes),
            "task_source_hashes": sum(not key.startswith("trace:") for key in hashes),
            "task_016_offline_workspaces": 9,
            "synthetic_oracle_controls": len(cases),
            "deterministic_oracle_checks_passed": len(oracle_checks) - len(failed),
            "deterministic_oracle_checks_total": len(oracle_checks),
            "oracle_control_scores": {key: value["score"] for key, value in oracle_scores.items()},
        }
        if failed:
            console.print({**summary, "failed_checks": failed})
            raise typer.Exit(code=2)
        console.print(summary)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"Outcome preflight failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=2) from None


@app.command("outcome-report")
def outcome_report(
    config: Path = typer.Option(Path("configs/phase1.corrected.pilot-v14.yaml"), exists=True),
    source_database: Path = typer.Option(Path("data/phase1/corrected/pilot-v14/runs.sqlite"), exists=True),
    judge_database: Path = typer.Option(OUTCOME_DIR / "judgments.sqlite", exists=True),
    rubric: Path = typer.Option(OUTCOME_RUBRIC, exists=True),
    calibration: Path = typer.Option(OUTCOME_DIR / "calibration.json", exists=True),
    output_dir: Path = typer.Option(OUTCOME_DIR / "report"),
    experiment_id: str = typer.Option("phase1_corrected_pilot_v14"),
    seed: int = typer.Option(1701),
) -> None:
    """Validate frozen v7 judgments and build provisional tables, review packet, and PDF."""
    experiment_config = ExperimentConfig.load(config)
    try:
        private_values = load_env_file(
            experiment_config.root / experiment_config.inference.get("env_file", ".env.uni-gpu.local")
        )
        pinned = json.loads(
            (experiment_config.root / experiment_config.inference["manifest"]).read_text(encoding="utf-8")
        )
    except Exception as exc:
        console.print(f"Outcome report failed: {type(exc).__name__}: required local inference manifest is unavailable")
        raise typer.Exit(code=2) from None
    secrets = tuple(
        value
        for value in (
            private_values.get("INDIC_UNI_GPU_BASE_URL"),
            private_values.get("INDIC_UNI_GPU_API_KEY"),
            pinned.get("resolved_model"),
        )
        if isinstance(value, str) and value
    )
    try:
        source = experiment_config.root / source_database
        result = build_outcome_report_v2(
            root=experiment_config.root,
            source_database=source,
            judge_database=experiment_config.root / judge_database,
            calibration_path=experiment_config.root / calibration,
            rubric_path=experiment_config.root / rubric,
            output_dir=experiment_config.root / output_dir,
            experiment_id=experiment_id,
            dataset_manifest=experiment_config.experiment["dataset_manifest"],
            secrets=secrets,
            seed=seed,
        )
        pdf = experiment_config.root / "output/pdf/indic_harness_phase1_llm_judged_regrade_v7.pdf"
        build_outcome_pdf_v2(result, pdf)
        _scan_for_secrets([pdf], secrets)
        verify_frozen_inputs(experiment_config.root, source)
        console.print(f"Wrote provisional outcome report: {result['counts']}")
        console.print(f"PDF: {pdf}")
    except Exception as exc:
        message = redact_text(str(exc), secrets)
        console.print(f"Outcome report failed: {type(exc).__name__}: {message}")
        raise typer.Exit(code=2) from None


@app.command("calibrate-outcomes-v8")
def calibrate_v8_command(
    config: Path = typer.Option(Path("configs/phase1.corrected.pilot-v14.yaml"), exists=True),
    source_database: Path = typer.Option(
        Path("data/phase1/corrected/pilot-v14/runs.sqlite"), exists=True
    ),
    rubric: Path = typer.Option(OUTCOME_V8_RUBRIC, exists=True),
    output: Path = typer.Option(OUTCOME_V8_DIR),
) -> None:
    """Preflight saved workspaces and run the frozen 39-control v8 calibration."""
    runtime = None
    try:
        experiment = ExperimentConfig.load(config)
        runtime = OutcomeJudgeRuntime(experiment.root, experiment.inference)
        with runtime:
            result = calibrate_outcomes_v8(
                experiment.root,
                experiment.root / source_database,
                experiment.data,
                experiment.root / rubric,
                experiment.root / output,
                runtime,
            )
        console.print(
            {
                "status": result["status"],
                "controls": result["control_count"],
                "failed_checks": result["checks_failed"],
            }
        )
    except Exception as exc:
        _safe_command_error("V8 calibration", exc, _runtime_secret_mapping(runtime))
        raise typer.Exit(code=2) from None


@app.command("judge-outcomes-v8")
def judge_v8_command(
    config: Path = typer.Option(Path("configs/phase1.corrected.pilot-v14.yaml"), exists=True),
    source_database: Path = typer.Option(
        Path("data/phase1/corrected/pilot-v14/runs.sqlite"), exists=True
    ),
    rubric: Path = typer.Option(OUTCOME_V8_RUBRIC, exists=True),
    output: Path = typer.Option(OUTCOME_V8_DIR),
) -> None:
    """Judge only the 44 archived v14 workspaces; never rerun agents."""
    runtime = None
    try:
        experiment = ExperimentConfig.load(config)
        runtime = OutcomeJudgeRuntime(experiment.root, experiment.inference)
        with runtime:
            result = rejudge_outcomes_v8(
                experiment.root,
                experiment.root / source_database,
                experiment.data,
                experiment.root / rubric,
                experiment.root / output,
                runtime,
            )
        console.print(result)
    except Exception as exc:
        _safe_command_error("V8 rejudgment", exc, _runtime_secret_mapping(runtime))
        raise typer.Exit(code=2) from None


@app.command("outcome-report-v8")
def outcome_report_v8_command(
    config: Path = typer.Option(Path("configs/phase1.corrected.pilot-v14.yaml"), exists=True),
    source_database: Path = typer.Option(
        Path("data/phase1/corrected/pilot-v14/runs.sqlite"), exists=True
    ),
    rubric: Path = typer.Option(OUTCOME_V8_RUBRIC, exists=True),
    output: Path = typer.Option(OUTCOME_V8_DIR),
) -> None:
    """Create provisional v8 tables and PDF after full technical coverage."""
    secrets: tuple[str, ...] = ()
    try:
        experiment = ExperimentConfig.load(config)
        private = load_env_file(
            experiment.root / experiment.inference.get("env_file", ".env.uni-gpu.local")
        )
        pinned = json.loads(
            (experiment.root / experiment.inference["manifest"]).read_text(encoding="utf-8")
        )
        secrets = tuple(
            value for value in (
                private.get("INDIC_UNI_GPU_BASE_URL"),
                private.get("INDIC_UNI_GPU_API_KEY"),
                pinned.get("resolved_model"),
            ) if isinstance(value, str) and value
        )
        directory = experiment.root / output
        report = build_v8_tables(
            experiment.root,
            experiment.root / source_database,
            directory / "judgments.sqlite",
            directory / "calibration.json",
            experiment.root / rubric,
            directory / "report",
            str(experiment.sandbox["image"]),
            secrets,
        )
        pdf = experiment.root / "output/pdf/indic_harness_phase1_llm_judged_regrade_v8.pdf"
        build_v8_pdf(report, pdf)
        _scan_for_secrets([pdf], secrets)
        verify_frozen_inputs(experiment.root, experiment.root / source_database)
        console.print({"coverage": report["coverage"], "calibration": report["calibration_status"]})
        console.print(f"PDF: {pdf}")
    except Exception as exc:
        console.print(f"V8 report failed: {type(exc).__name__}: {redact_text(str(exc), secrets)}")
        raise typer.Exit(code=2) from None


if __name__ == "__main__":
    app()
