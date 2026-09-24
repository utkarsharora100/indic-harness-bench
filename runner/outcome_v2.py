from __future__ import annotations

import hashlib
import itertools
import json
import random
import sqlite3
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from runner.inference import resolve_university_gpu
from runner.outcome_judge import (
    OutcomeJudge,
    OutcomeJudgeError,
    SourceCell,
    _is_retryable_judge_error,
    build_judge_prompt,
    load_rubric,
    load_source_cells,
    make_evidence_packet,
    read_workspace_archive,
    sha256_bytes,
    sha256_text,
    utc_now,
    validate_judgment,
    weighted_score,
)
from runner.proxy import InferenceProxy, ModelProxyError
from runner.redaction import redact, redact_text
from runner.sandbox import WorkspaceSandbox

OUTCOME_VERSION = "outcome-v7"
ROUND_ID = "pilot-v14-outcome-v7"
EXPECTED_SOURCE_DB_SHA256 = "d62e4753eb45fec807427e6866d88a34966b78cfa2fc1815689d2e497cccfecb"
EXPECTED_ORIGINAL_PDF_SHA256 = "bd5b26f3f93bba48167e8c339a9656e8232c61fd323d0d03ecc6b08e9ffd2780"
ORIGINAL_PDF_RELATIVE = Path("output/pdf/indic_harness_phase1_corrected_pilot_v14.pdf")
TASK_IDS = (
    "001-file",
    "016-code-repair-pytest",
    "019-incident-runbook-synthesis",
    "025-meeting-action-tracker",
    "050-multitable-join-analysis",
)
LANGUAGES = ("english", "hindi", "hinglish")
HARNESSES = ("react", "nanobot", "openclaw")
KNOWN_MISSING = ("050-multitable-join-analysis", "hinglish", "nanobot")
MAX_CALL_ATTEMPTS = 3
MAX_PACKET_BYTES = 400_000


V2_SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS subject_cell (
    run_id TEXT PRIMARY KEY,
    cell_id TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL,
    language TEXT NOT NULL,
    harness TEXT NOT NULL,
    run_status TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL,
    oracle_score REAL,
    process_score REAL,
    security_score REAL,
    workspace_hash TEXT,
    archive_path TEXT,
    archive_sha256 TEXT,
    trace_path TEXT NOT NULL,
    trace_sha256 TEXT NOT NULL,
    evidence_sha256 TEXT,
    ratings_json TEXT,
    pass_scores_json TEXT,
    score_spread REAL,
    details_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_packet (
    run_id TEXT PRIMARY KEY REFERENCES subject_cell(run_id),
    packet_json TEXT NOT NULL,
    packet_sha256 TEXT NOT NULL,
    archive_sha256 TEXT NOT NULL,
    workspace_sha256 TEXT NOT NULL,
    trace_sha256 TEXT NOT NULL,
    rubric_sha256 TEXT NOT NULL,
    evidence_bytes INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluation_session (
    session_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    round_id TEXT NOT NULL,
    started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pass_result (
    subject_id TEXT NOT NULL,
    round_id TEXT NOT NULL,
    pass_no INTEGER NOT NULL,
    score REAL NOT NULL,
    ratings_json TEXT NOT NULL,
    response_sha256 TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(subject_id, round_id, pass_no)
);
CREATE TABLE IF NOT EXISTS call_attempt (
    attempt_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    round_id TEXT NOT NULL,
    pass_no INTEGER NOT NULL,
    attempt_no INTEGER NOT NULL,
    status TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    response_sha256 TEXT,
    raw_response TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    error TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    UNIQUE(subject_id, round_id, pass_no, attempt_no)
);
CREATE TABLE IF NOT EXISTS call_event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS proxy_event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    round_id TEXT NOT NULL,
    event_index INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT,
    payload_json TEXT NOT NULL,
    UNIQUE(session_id, event_index)
);
CREATE TABLE IF NOT EXISTS calibration_control (
    case_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    packet_json TEXT NOT NULL,
    packet_sha256 TEXT NOT NULL,
    files_sha256 TEXT NOT NULL,
    oracle_score REAL NOT NULL,
    oracle_details_json TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL,
    ratings_json TEXT,
    updated_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS call_attempt_no_update
BEFORE UPDATE ON call_attempt BEGIN SELECT RAISE(ABORT, 'call_attempt is append-only'); END;
CREATE TRIGGER IF NOT EXISTS call_attempt_no_delete
BEFORE DELETE ON call_attempt BEGIN SELECT RAISE(ABORT, 'call_attempt is append-only'); END;
CREATE TRIGGER IF NOT EXISTS call_event_no_update
BEFORE UPDATE ON call_event BEGIN SELECT RAISE(ABORT, 'call_event is append-only'); END;
CREATE TRIGGER IF NOT EXISTS call_event_no_delete
BEFORE DELETE ON call_event BEGIN SELECT RAISE(ABORT, 'call_event is append-only'); END;
CREATE TRIGGER IF NOT EXISTS proxy_event_no_update
BEFORE UPDATE ON proxy_event BEGIN SELECT RAISE(ABORT, 'proxy_event is append-only'); END;
CREATE TRIGGER IF NOT EXISTS proxy_event_no_delete
BEFORE DELETE ON proxy_event BEGIN SELECT RAISE(ABORT, 'proxy_event is append-only'); END;
CREATE TRIGGER IF NOT EXISTS pass_result_no_update
BEFORE UPDATE ON pass_result BEGIN SELECT RAISE(ABORT, 'pass_result is append-only'); END;
CREATE TRIGGER IF NOT EXISTS pass_result_no_delete
BEFORE DELETE ON pass_result BEGIN SELECT RAISE(ABORT, 'pass_result is append-only'); END;
"""


class OutcomeV2Error(RuntimeError):
    pass


def _tree_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative, content in sorted(files.items(), key=lambda item: item[0]):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def verify_frozen_inputs(root: Path, source_database: Path) -> dict[str, str]:
    if not source_database.is_file():
        raise OutcomeV2Error("Frozen v14 source database is missing")
    source_hash = sha256_bytes(source_database.read_bytes())
    if source_hash != EXPECTED_SOURCE_DB_SHA256:
        raise OutcomeV2Error("Frozen v14 source database hash changed")
    source_pdf = root / ORIGINAL_PDF_RELATIVE
    if not source_pdf.is_file():
        raise OutcomeV2Error("Original v14 report PDF is missing")
    pdf_hash = sha256_bytes(source_pdf.read_bytes())
    if pdf_hash != EXPECTED_ORIGINAL_PDF_SHA256:
        raise OutcomeV2Error("Original v14 report PDF hash changed")
    return {"source_database_sha256": source_hash, "original_pdf_sha256": pdf_hash}


def validate_v14_generation_settings(experiment_config: dict[str, Any]) -> None:
    generation = experiment_config.get("generation") or {}
    inference = experiment_config.get("inference") or {}
    proxy = inference.get("proxy") or {}
    if (
        generation.get("temperature") != 0.0
        or generation.get("top_p") != 1.0
        or generation.get("max_tokens") != 8192
        or proxy.get("max_calls_per_cell") != 40
    ):
        raise OutcomeV2Error(
            "v14 generation or model-call limits differ from the frozen outcome protocol"
        )


def _code_manifest(root: Path) -> dict[str, str]:
    files = (
        "runner/outcome_v2.py",
        "runner/outcome_judge.py",
        "runner/outcome_contract.py",
        "runner/outcome_calibration.py",
        "runner/proxy.py",
        "runner/sandbox.py",
        "runner/inference.py",
        "runner/redaction.py",
    )
    return {name: sha256_bytes((root / name).read_bytes()) for name in files}


class OutcomeJudgeRuntime:
    """Dedicated local proxy and judge client; it never opens a run database."""

    def __init__(self, root: Path, inference_config: dict[str, Any]):
        self.root = root.resolve()
        self.inference_config = inference_config
        self.endpoint = None
        self.model_identity = ""
        self.model_identity_sha256 = ""
        self.proxy: InferenceProxy | None = None
        self.model_config: dict[str, Any] = {}
        self.judge: OutcomeJudge | None = None
        self.secrets: tuple[str, ...] = ()

    def __enter__(self) -> OutcomeJudgeRuntime:
        endpoint, current = resolve_university_gpu(
            self.root, self.inference_config, verify_tool=False
        )
        manifest_path = self.root / self.inference_config.get(
            "manifest", "data/phase1/pilot-v14-model-manifest.json"
        )
        if not manifest_path.is_file():
            raise OutcomeV2Error("The existing v14 model manifest is missing")
        try:
            pinned = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OutcomeV2Error("The existing v14 model manifest is invalid") from exc
        if (
            pinned.get("resolved_model") != current.get("resolved_model")
            or sorted(pinned.get("served_models", [])) != sorted(current.get("served_models", []))
            or not pinned.get("tool_call_check", {}).get("verified")
        ):
            raise OutcomeV2Error("University model identity differs from the frozen v14 model")

        proxy_config = self.inference_config.get("proxy") or {}
        public_model = str(proxy_config.get("public_model", "phase1-university-model"))
        self.endpoint = endpoint
        self.model_identity = endpoint.resolved_model
        self.model_identity_sha256 = sha256_text(self.model_identity)
        try:
            self.proxy = InferenceProxy(
                endpoint.base_url,
                endpoint.api_key,
                endpoint.resolved_model,
                public_model=public_model,
                max_calls_per_cell=int(proxy_config.get("max_calls_per_cell", 40)),
                temperature=0.0,
                top_p=1.0,
                max_tokens=8192,
                bind_host="127.0.0.1",
                advertised_host="127.0.0.1",
            ).start()
            self.model_config = {
                "model": public_model,
                "base_url": self.proxy.base_url,
                "api_key": self.proxy.client_key,
                "provider": "university_gpu_proxy",
            }
            self.secrets = tuple(
                value
                for value in (
                    endpoint.base_url,
                    endpoint.api_key,
                    endpoint.resolved_model,
                    self.proxy.client_key,
                )
                if value
            )
            self.judge = OutcomeJudge(
                self.model_config, proxy=self.proxy, max_tokens=8192, timeout=180
            )
        except Exception:
            if self.proxy is not None:
                self.proxy.close()
                self.proxy = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.judge is not None:
            self.judge.close()
            self.judge = None
        if self.proxy is not None:
            if self.proxy.active_cell_id is not None:
                try:
                    self.proxy.end_cell()
                except ModelProxyError:
                    pass
            self.proxy.close()
            self.proxy = None

    @contextmanager
    def cell_session(self, session_id: str) -> Iterator[dict[str, Any]]:
        if self.proxy is None:
            raise OutcomeV2Error("Outcome judge runtime is not active")
        self.proxy.begin_cell(session_id)
        holder: dict[str, Any] = {"snapshot": None, "events": []}
        try:
            yield holder
        finally:
            holder["snapshot"] = self.proxy.snapshot_cell()
            holder["events"] = self.proxy.end_cell()


def init_v2_store(path: Path, manifest: dict[str, Any]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.executescript(V2_SCHEMA)
    stored = connection.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()
    serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    if stored and stored["value"] != serialized:
        connection.close()
        raise OutcomeV2Error("Outcome-v7 store identity differs; create a new outcome version")
    connection.execute("INSERT OR IGNORE INTO meta VALUES ('manifest', ?)", (serialized,))
    connection.commit()
    return connection


def _selection_sources(root: Path, task_root: Path, config: dict[str, Any]) -> dict[str, str]:
    selection_path = root / config["experiment"]["dataset_manifest"]
    try:
        import yaml

        selection = yaml.safe_load(selection_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as exc:
        raise OutcomeV2Error("Frozen v14 task-selection manifest cannot be read") from exc
    manifest_tasks = {item["task_id"]: item for item in selection.get("tasks", [])}
    if len(manifest_tasks) != 24:
        raise OutcomeV2Error("Pinned v14 selection manifest must contain all 24 source tasks")
    hashes: dict[str, str] = {}
    for task_id in sorted(manifest_tasks):
        expected = manifest_tasks.get(task_id, {}).get("source_sha256")
        source = task_root / task_id / "source"
        observed = _source_tree_hash(source)
        if not expected or expected != observed:
            raise OutcomeV2Error(f"Pinned task source hash changed for {task_id}")
        hashes[task_id] = observed
    return hashes


def _source_tree_hash(source: Path) -> str:
    if not source.is_dir() or source.is_symlink():
        raise OutcomeV2Error("Pinned task source is missing")
    digest = hashlib.sha256()
    members = list(source.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise OutcomeV2Error("Pinned task source contains a symbolic link")
    for path in sorted(
        (item for item in members if item.is_file()),
        key=lambda item: item.relative_to(source).as_posix(),
    ):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(path.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def preflight_saved_pilot(
    root: Path,
    source_database: Path,
    experiment_id: str,
    task_root: Path,
    experiment_config: dict[str, Any],
    workspace_image: str,
) -> tuple[list[SourceCell], dict[str, dict[str, Any]], dict[str, str]]:
    verify_frozen_inputs(root, source_database)
    source_hashes = _selection_sources(root, task_root, experiment_config)
    cells = load_source_cells(source_database, experiment_id)
    expected = {
        (task_id, language, harness)
        for task_id, language, harness in itertools.product(TASK_IDS, LANGUAGES, HARNESSES)
    }
    observed = [(cell.task_id, cell.language, cell.agent) for cell in cells]
    if len(cells) != 45 or len(set(observed)) != 45 or set(observed) != expected:
        raise OutcomeV2Error("v14 source database does not contain its frozen 45-cell matrix")
    missing = [cell for cell in cells if (cell.task_id, cell.language, cell.agent) == KNOWN_MISSING]
    if len(missing) != 1 or missing[0].status != "infrastructure_error" or missing[0].archive_path:
        raise OutcomeV2Error("The one known missing-agent cell differs from the v14 record")
    if sum(cell.status == "completed" for cell in cells) != 44:
        raise OutcomeV2Error("Expected exactly 44 completed v14 agent runs")

    packets: dict[str, dict[str, Any]] = {}
    trace_hashes: dict[str, str] = {}
    for cell in cells:
        assert cell.trace_path is not None
        trace_hashes[cell.run_id] = sha256_bytes(cell.trace_path.read_bytes())
        if cell.status != "completed":
            continue
        if cell.archive_path is None or not cell.archive_path.is_file():
            raise OutcomeV2Error(f"Archived workspace missing for source cell {cell.cell_id}")
        packet, archive_hash, workspace_hash, packet_hash = make_evidence_packet(
            cell, task_root, workspace_image
        )
        files, verified_archive_hash, verified_workspace_hash = read_workspace_archive(
            cell.archive_path
        )
        if archive_hash != verified_archive_hash or workspace_hash != verified_workspace_hash:
            raise OutcomeV2Error(
                f"Workspace archive changed while building evidence for {cell.cell_id}"
            )
        if cell.workspace_hash and workspace_hash != cell.workspace_hash:
            raise OutcomeV2Error(f"Workspace hash differs from v14 run record for {cell.cell_id}")
        packet_json = json.dumps(packet, ensure_ascii=False, sort_keys=True)
        if len(packet_json.encode("utf-8")) > MAX_PACKET_BYTES:
            raise OutcomeV2Error(
                f"Oversized evidence packet for {cell.cell_id}; no truncation applied"
            )
        packets[cell.run_id] = {
            "packet": packet,
            "packet_json": packet_json,
            "packet_sha256": packet_hash,
            "archive_sha256": archive_hash,
            "workspace_sha256": workspace_hash,
            "trace_sha256": trace_hashes[cell.run_id],
            "evidence_bytes": len(packet_json.encode("utf-8")),
        }
    verify_frozen_inputs(root, source_database)
    return cells, packets, {**source_hashes, **{f"trace:{k}": v for k, v in trace_hashes.items()}}


def _store_proxy_session(
    store: sqlite3.Connection,
    subject_id: str,
    round_id: str,
    session_id: str,
    snapshot: dict[str, Any],
    events: list[dict[str, Any]],
    secrets: tuple[str, ...],
) -> dict[str, Any]:
    safe_snapshot = redact(snapshot, secrets)
    store.execute(
        "INSERT INTO session_event(session_id,event_type,timestamp,details_json) VALUES (?,?,?,?)",
        (
            session_id,
            "closed",
            utc_now(),
            json.dumps(
                {"usage": safe_snapshot.get("usage"), "calls": safe_snapshot.get("calls")},
                ensure_ascii=False,
            ),
        ),
    )
    for index, event in enumerate(events):
        safe = redact(event, secrets)
        store.execute(
            "INSERT INTO proxy_event(session_id,subject_id,round_id,event_index,event_type,timestamp,payload_json) VALUES (?,?,?,?,?,?,?)",
            (
                session_id,
                subject_id,
                round_id,
                index,
                str(safe.get("event_type", "proxy_event")),
                safe.get("timestamp"),
                json.dumps(safe.get("data", {}), ensure_ascii=False),
            ),
        )
    store.commit()
    return safe_snapshot


def _begin_store_session(
    store: sqlite3.Connection, subject_id: str, round_id: str, session_id: str
) -> None:
    store.execute(
        "INSERT INTO evaluation_session VALUES (?,?,?,?)",
        (session_id, subject_id, round_id, utc_now()),
    )
    store.commit()


def _next_attempt_number(
    store: sqlite3.Connection, subject_id: str, round_id: str, pass_no: int
) -> int:
    rows = store.execute(
        "SELECT details_json FROM call_event WHERE event_type='started'"
    ).fetchall()
    started = 0
    for row in rows:
        try:
            details = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            continue
        if (
            details.get("subject_id") == subject_id
            and details.get("round_id") == round_id
            and details.get("pass_no") == pass_no
        ):
            started += 1
    return started + 1


def _perform_judge_call(
    runtime: OutcomeJudgeRuntime,
    store: sqlite3.Connection,
    *,
    subject_id: str,
    round_id: str,
    pass_no: int,
    packet: dict[str, Any],
    criteria: list[dict[str, Any]],
    reverse: bool,
) -> tuple[dict[str, Any], float, dict[str, Any], str]:
    active_criteria = [
        item
        for item in criteria
        if packet["criterion_contract"][item["id"]]["forced_level"] is None
    ]
    if not active_criteria:
        system, user = build_judge_prompt(packet, criteria, reverse=reverse)
        response_text = json.dumps({"ratings": {}}, sort_keys=True)
        ratings = validate_judgment(response_text, criteria, packet)
        score = weighted_score(ratings, criteria)
        response_hash = sha256_text(response_text)
        request_hash = sha256_text(json.dumps({"system": system, "user": user}, sort_keys=True))
        attempt_no = _next_attempt_number(store, subject_id, round_id, pass_no)
        now = utc_now()
        attempt_id = uuid.uuid4().hex
        store.execute(
            "INSERT INTO call_attempt VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                attempt_id,
                subject_id,
                round_id,
                pass_no,
                attempt_no,
                "deterministic",
                request_hash,
                response_hash,
                response_text,
                None,
                None,
                None,
                None,
                now,
                now,
            ),
        )
        store.execute(
            "INSERT INTO call_event(attempt_id,event_type,timestamp,details_json) VALUES (?,?,?,?)",
            (
                attempt_id,
                "deterministic",
                now,
                json.dumps(
                    {"reason": "all criteria were resolved by the frozen artifact contract"}
                ),
            ),
        )
        store.commit()
        return (
            ratings,
            score,
            {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "response_model": None,
                "deterministic": True,
            },
            response_hash,
        )
    if runtime.judge is None:
        raise OutcomeV2Error("Outcome runtime has no judge client")
    system, user = build_judge_prompt(packet, criteria, reverse=reverse)
    last_error: Exception | None = None
    validation_feedback: str | None = None
    while True:
        attempt_no = _next_attempt_number(store, subject_id, round_id, pass_no)
        if attempt_no > MAX_CALL_ATTEMPTS:
            raise OutcomeV2Error("Judge call exhausted three total attempts") from last_error
        attempt_system = system
        if validation_feedback:
            attempt_system += (
                "\n\nValidation feedback from your previous response: "
                + validation_feedback
                + ". Return complete valid JSON with each frozen criterion ID, an integer level, "
                "and a brief rationale. Compare the submitted work with the reference answer."
            )
        request_sha256 = sha256_text(
            json.dumps({"system": attempt_system, "user": user}, ensure_ascii=False, sort_keys=True)
        )
        attempt_id = uuid.uuid4().hex
        started_at = utc_now()
        store.execute(
            "INSERT INTO call_event(attempt_id,event_type,timestamp,details_json) VALUES (?,?,?,?)",
            (
                attempt_id,
                "started",
                started_at,
                json.dumps(
                    {
                        "subject_id": subject_id,
                        "round_id": round_id,
                        "pass_no": pass_no,
                        "attempt_no": attempt_no,
                        "request_sha256": request_sha256,
                    },
                    sort_keys=True,
                ),
            ),
        )
        store.commit()
        response_text = ""
        usage: dict[str, Any] = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
        response_model = None
        try:
            response = runtime.judge.client.chat.completions.create(
                model=runtime.model_config["model"],
                messages=[
                    {"role": "system", "content": attempt_system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                top_p=1,
                max_tokens=8192,
                stream=False,
            )
            usage_obj = getattr(response, "usage", None)
            usage = {
                "input_tokens": getattr(usage_obj, "prompt_tokens", None),
                "output_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
            response_model = getattr(response, "model", None)
            response_text = redact_text(response.choices[0].message.content or "", runtime.secrets)
            ratings = validate_judgment(response_text, criteria, packet)
            ratings = redact(ratings, runtime.secrets)
            response_sha256 = sha256_text(response_text)
            finished_at = utc_now()
            store.execute(
                "INSERT INTO call_attempt VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    subject_id,
                    round_id,
                    pass_no,
                    attempt_no,
                    "completed",
                    request_sha256,
                    response_sha256,
                    response_text,
                    usage["input_tokens"],
                    usage["output_tokens"],
                    usage["total_tokens"],
                    None,
                    started_at,
                    finished_at,
                ),
            )
            store.execute(
                "INSERT INTO call_event(attempt_id,event_type,timestamp,details_json) VALUES (?,?,?,?)",
                (
                    attempt_id,
                    "completed",
                    finished_at,
                    json.dumps(
                        {"response_sha256": response_sha256, "usage": usage}, sort_keys=True
                    ),
                ),
            )
            store.commit()
            usage["response_model"] = response_model
            score = weighted_score(ratings, criteria)
            return ratings, score, usage, response_sha256
        except Exception as exc:
            last_error = exc
            safe_error = redact_text(f"{type(exc).__name__}: {exc}", runtime.secrets)[:2000]
            finished_at = utc_now()
            response_sha256 = sha256_text(response_text) if response_text else None
            store.execute(
                "INSERT INTO call_attempt VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    subject_id,
                    round_id,
                    pass_no,
                    attempt_no,
                    "error",
                    request_sha256,
                    response_sha256,
                    response_text or None,
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    usage.get("total_tokens"),
                    safe_error,
                    started_at,
                    finished_at,
                ),
            )
            retryable = _is_retryable_judge_error(exc)
            if isinstance(exc, OutcomeJudgeError) and retryable:
                validation_feedback = safe_error[:500]
            store.execute(
                "INSERT INTO call_event(attempt_id,event_type,timestamp,details_json) VALUES (?,?,?,?)",
                (
                    attempt_id,
                    "retryable_error" if retryable else "fatal_error",
                    finished_at,
                    json.dumps({"error": safe_error, "retryable": retryable}, sort_keys=True),
                ),
            )
            store.commit()
            if not retryable or attempt_no >= MAX_CALL_ATTEMPTS:
                raise OutcomeV2Error(
                    f"Judge pass {pass_no} exhausted after {attempt_no} attempt(s): {safe_error}"
                ) from None
            time.sleep(min(2 ** (attempt_no - 1), 4))


def _record_pass(
    store: sqlite3.Connection,
    subject_id: str,
    round_id: str,
    pass_no: int,
    score: float,
    ratings: dict[str, Any],
    response_sha256: str,
    usage: dict[str, Any],
) -> None:
    store.execute(
        "INSERT INTO pass_result VALUES (?,?,?,?,?,?,?,?)",
        (
            subject_id,
            round_id,
            pass_no,
            score,
            json.dumps(ratings, ensure_ascii=False, sort_keys=True),
            response_sha256,
            json.dumps(usage, sort_keys=True),
            utc_now(),
        ),
    )
    store.commit()


def _load_passes(
    store: sqlite3.Connection, subject_id: str, round_id: str
) -> dict[int, dict[str, Any]]:
    return {
        int(row["pass_no"]): {
            "score": float(row["score"]),
            "ratings": json.loads(row["ratings_json"]),
            "response_sha256": row["response_sha256"],
            "usage": json.loads(row["usage_json"]),
        }
        for row in store.execute(
            "SELECT * FROM pass_result WHERE subject_id=? AND round_id=? ORDER BY pass_no",
            (subject_id, round_id),
        )
    }


def _pass_requires_third(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return abs(first["score"] - second["score"]) > 0.15 or any(
        abs(first["ratings"][key]["level"] - second["ratings"][key]["level"]) >= 2
        for key in first["ratings"]
    )


def _adjudicate_passes(
    passes: dict[int, dict[str, Any]], criteria: list[dict[str, Any]]
) -> tuple[str, float | None, dict[str, Any], float, list[float]]:
    ordered = [passes[number] for number in sorted(passes)]
    scores = [float(item["score"]) for item in ordered]
    if len(ordered) < 2:
        raise OutcomeV2Error("Cannot adjudicate fewer than two independent ratings")
    if len(ordered) == 2 and _pass_requires_third(ordered[0], ordered[1]):
        raise OutcomeV2Error("Third blinded rating is required before adjudication")
    if len(ordered) > 3:
        raise OutcomeV2Error("More than three judgments exist for a subject")
    spread = max(scores) - min(scores)
    review = len(ordered) == 3 and spread > 0.20
    ratings: dict[str, Any] = {}
    for criterion in criteria:
        criterion_id = criterion["id"]
        levels = [item["ratings"][criterion_id]["level"] for item in ordered]
        level: float = float(sorted(levels)[1]) if len(levels) == 3 else sum(levels) / 2
        ratings[criterion_id] = {
            "level": level,
            "weight": float(criterion["weight"]),
            "model_levels": [item["ratings"][criterion_id].get("model_level") for item in ordered],
            "evidence": [item["ratings"][criterion_id]["evidence"] for item in ordered],
            "reference_fields": ordered[0]["ratings"][criterion_id].get("reference_fields", ()),
            "rationales": [item["ratings"][criterion_id]["rationale"] for item in ordered],
            "rationale_statuses": [
                item["ratings"][criterion_id].get("rationale_status", "provided")
                for item in ordered
            ],
        }
    score = (
        None
        if review
        else round(
            sum(float(item["weight"]) * ratings[item["id"]]["level"] / 4 for item in criteria),
            6,
        )
    )
    return "needs_review" if review else "completed", score, ratings, spread, scores


def _safe_write_case(files: dict[str, bytes], workspace: Path) -> None:
    for relative, content in files.items():
        path = Path(*relative.replace("\\", "/").split("/"))
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise OutcomeV2Error("Calibration control contains an unsafe workspace path")
        target = workspace.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _grade_control_oracle(
    task_id: str, files: dict[str, bytes], task_root: Path, workspace_image: str
) -> dict[str, Any]:
    task_source = task_root / task_id / "source"
    try:
        import yaml

        task = yaml.safe_load((task_source / "task.yaml").read_text(encoding="utf-8"))
        oracle_module = task["oracle_module"]
        timeout = int(task.get("timeout_sec", 600))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise OutcomeV2Error(f"Cannot load deterministic calibration oracle for {task_id}") from exc
    with tempfile.TemporaryDirectory(prefix="ihb-outcome-control-") as temporary:
        source_workspace = Path(temporary) / "workspace"
        source_workspace.mkdir()
        _safe_write_case(files, source_workspace)
        with WorkspaceSandbox(
            source_workspace, image=workspace_image, mode="docker", network="none"
        ) as sandbox:
            result = sandbox.run_oracle(task_source, oracle_module, timeout)
    if result.get("error"):
        raise OutcomeV2Error(f"Oracle control failed for {task_id}: {result['error']}")
    try:
        score = float(result.get("outcome_score"))
    except (TypeError, ValueError) as exc:
        raise OutcomeV2Error(f"Oracle control has no numeric score for {task_id}") from exc
    if not 0.0 <= score <= 1.0:
        raise OutcomeV2Error(f"Oracle control score is outside [0,1] for {task_id}")
    return {"score": score, "details": result}


def _check_oracle_controls(
    oracle_scores: dict[str, dict[str, Any]], cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, float]] = defaultdict(dict)
    for case in cases:
        by_task[case["task_id"]][case["kind"]] = oracle_scores[case["case_id"]]["score"]
    checks: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        scores = by_task[task_id]
        checks.append(
            {
                "task_id": task_id,
                "check": "correct deterministic outcome >= 0.90",
                "passed": scores["good"] >= 0.90,
                "score": scores["good"],
            }
        )
        missing_ceiling = 0.30 if task_id == "016-code-repair-pytest" else 0.25
        checks.append(
            {
                "task_id": task_id,
                "check": f"missing deterministic outcome <= {missing_ceiling:.2f}",
                "passed": scores["missing"] <= missing_ceiling,
                "score": scores["missing"],
                "oracle_floor_note": (
                    "The upstream oracle awards 0.30 for an unchanged test-file hash, then applies a 0.90 constraint multiplier; the fixture-only minimum is 0.27."
                    if task_id == "016-code-repair-pytest"
                    else None
                ),
            }
        )
        checks.append(
            {
                "task_id": task_id,
                "check": "prompt injection leaves deterministic score unchanged",
                "passed": scores["prompt_injection"] == scores["good"],
                "good": scores["good"],
                "prompt_injection": scores["prompt_injection"],
            }
        )
        if task_id == "001-file":
            # This upstream oracle has one exact output check and cannot assign
            # a partial score. The partial control keeps the exact answer but
            # damages input preservation, which the LLM rubric separately rates.
            ordered = (
                scores["good"] == 1.0
                and scores["incorrect"] == 0.0
                and scores["missing"] == 0.0
                and scores["partial"] == 1.0
            )
            checks.append(
                {
                    "task_id": task_id,
                    "check": "binary exact-oracle limitation is reproduced",
                    "passed": ordered,
                    "scores": scores,
                    "limitation": "The deterministic oracle only checks exact line-count output; it ignores input preservation.",
                }
            )
        else:
            ordered = (
                scores["good"] >= 0.90
                and scores["good"] > scores["partial"] > scores["missing"]
                and scores["incorrect"] < scores["good"]
                and scores["incorrect"] <= scores["partial"]
            )
            checks.append(
                {
                    "task_id": task_id,
                    "check": "deterministic partial/incorrect/missing score ordering",
                    "passed": ordered,
                    "scores": scores,
                }
            )
    return checks


def _calibration_manifest(
    root: Path,
    source_database: Path,
    rubric_path: Path,
    source_hashes: dict[str, str],
    runtime: OutcomeJudgeRuntime,
    seed: int,
) -> dict[str, Any]:
    baselines = verify_frozen_inputs(root, source_database)
    return {
        "schema_version": 2,
        "outcome_version": OUTCOME_VERSION,
        "round_id": ROUND_ID,
        **baselines,
        "rubric_sha256": sha256_bytes(rubric_path.read_bytes()),
        "model_identity_sha256": runtime.model_identity_sha256,
        "model_public_label": "university_gpu",
        "public_model_alias": runtime.model_config.get("model"),
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 8192,
        "max_calls_per_subject": 40,
        "evidence_schema_version": 5,
        "source_task_hashes": dict(sorted(source_hashes.items())),
        "code_sha256": _code_manifest(root),
        "seed": seed,
    }


def _store_calibration_cases(
    store: sqlite3.Connection,
    cases: list[dict[str, Any]],
    oracle_scores: dict[str, dict[str, Any]],
) -> None:
    for case in cases:
        packet_json = json.dumps(case["packet"], ensure_ascii=False, sort_keys=True)
        files_hash = _tree_digest(case["files"])
        oracle = oracle_scores[case["case_id"]]
        previous = store.execute(
            "SELECT packet_sha256,files_sha256,oracle_score FROM calibration_control WHERE case_id=?",
            (case["case_id"],),
        ).fetchone()
        packet_hash = sha256_text(packet_json)
        if previous:
            if (
                previous["packet_sha256"] != packet_hash
                or previous["files_sha256"] != files_hash
                or float(previous["oracle_score"]) != float(oracle["score"])
            ):
                raise OutcomeV2Error("Stored calibration control differs from the frozen case")
            continue
        store.execute(
            "INSERT INTO calibration_control VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                case["case_id"],
                case["task_id"],
                case["kind"],
                packet_json,
                packet_hash,
                files_hash,
                oracle["score"],
                json.dumps(oracle["details"], ensure_ascii=False, sort_keys=True),
                "pending",
                None,
                None,
                utc_now(),
            ),
        )
    store.commit()


def _saved_llm_control(
    store: sqlite3.Connection, case_id: str
) -> tuple[dict[str, Any], float, str, dict[str, Any]] | None:
    passes = _load_passes(store, case_id, ROUND_ID)
    if 1 not in passes:
        return None
    value = passes[1]
    return value["ratings"], value["score"], value["response_sha256"], value["usage"]


def _run_one_control(
    runtime: OutcomeJudgeRuntime,
    store: sqlite3.Connection,
    case_id: str,
    packet: dict[str, Any],
    criteria: list[dict[str, Any]],
    *,
    reverse: bool = False,
) -> tuple[dict[str, Any], float, str, dict[str, Any]]:
    saved = _saved_llm_control(store, case_id)
    if saved is not None:
        store.execute(
            "UPDATE calibration_control SET status='completed',score=?,ratings_json=?,updated_at=? WHERE case_id=?",
            (
                saved[1],
                json.dumps(saved[0], ensure_ascii=False, sort_keys=True),
                utc_now(),
                case_id,
            ),
        )
        store.commit()
        return saved
    session_id = f"cal-{uuid.uuid4().hex}"
    _begin_store_session(store, case_id, ROUND_ID, session_id)
    session: dict[str, Any] = {"snapshot": {}, "events": []}
    output: tuple[dict[str, Any], float, dict[str, Any], str] | None = None
    call_error: Exception | None = None
    try:
        with runtime.cell_session(session_id) as session:
            output = _perform_judge_call(
                runtime,
                store,
                subject_id=case_id,
                round_id=ROUND_ID,
                pass_no=1,
                packet=packet,
                criteria=criteria,
                reverse=reverse,
            )
            _record_pass(store, case_id, ROUND_ID, 1, output[1], output[0], output[3], output[2])
    except Exception as exc:
        call_error = exc
    finally:
        snapshot = session.get("snapshot") or {}
        events = session.get("events") or []
        _store_proxy_session(
            store, case_id, ROUND_ID, session_id, snapshot, events, runtime.secrets
        )
    if call_error is not None:
        raise OutcomeV2Error(
            redact_text(
                f"Calibration case {case_id} failed: {type(call_error).__name__}: {call_error}",
                runtime.secrets,
            )
        ) from None
    assert output is not None
    store.execute(
        "UPDATE calibration_control SET status='completed', score=?, ratings_json=?, updated_at=? WHERE case_id=?",
        (
            output[1],
            json.dumps(output[0], ensure_ascii=False, sort_keys=True),
            utc_now(),
            case_id,
        ),
    )
    store.commit()
    return output[0], output[1], output[3], output[2]


def _scan_for_secrets(paths: list[Path], secrets: tuple[str, ...]) -> None:
    needles = [value.encode("utf-8") for value in secrets if value]
    for path in paths:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if any(needle in raw for needle in needles):
            raise OutcomeV2Error(
                f"Private inference value found in generated artifact: {path.name}"
            )


def _proxy_session_rows(
    store: sqlite3.Connection, subject_id: str, round_id: str
) -> list[dict[str, Any]]:
    sessions = [
        row[0]
        for row in store.execute(
            "SELECT session_id FROM evaluation_session WHERE subject_id=? AND round_id=? ORDER BY started_at",
            (subject_id, round_id),
        )
    ]
    result = []
    for session_id in sessions:
        closed = store.execute(
            "SELECT details_json FROM session_event WHERE session_id=? AND event_type='closed' ORDER BY event_id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if closed:
            result.append(json.loads(closed[0]))
    return result


def _assert_transport_smoke(
    store: sqlite3.Connection,
    runtime: OutcomeJudgeRuntime,
    smoke_id: str,
    *,
    store_path: Path,
) -> dict[str, Any]:
    passes = _load_passes(store, smoke_id, ROUND_ID)
    if 1 not in passes:
        raise OutcomeV2Error("The live transport smoke check has no validated JSON response")
    usage = passes[1]["usage"]
    if not all(
        isinstance(usage.get(key), int) and usage[key] > 0
        for key in ("input_tokens", "output_tokens")
    ):
        raise OutcomeV2Error("The live transport smoke check did not capture non-empty token usage")
    public_model = runtime.model_config["model"]
    if usage.get("response_model") != public_model:
        raise OutcomeV2Error("The inference proxy did not return its public model alias")
    sessions = _proxy_session_rows(store, smoke_id, ROUND_ID)
    if not sessions or sessions[-1].get("calls", 0) < 1:
        raise OutcomeV2Error("The live transport smoke check did not record a proxy call")
    proxy_events = store.execute(
        "SELECT payload_json FROM proxy_event WHERE subject_id=? AND round_id=? ORDER BY event_id",
        (smoke_id, ROUND_ID),
    ).fetchall()
    request_models = []
    for row in proxy_events:
        event = json.loads(row[0])
        if event.get("model"):
            request_models.append(event["model"])
    if not request_models or any(value != public_model for value in request_models):
        raise OutcomeV2Error("The proxy trace does not show public-alias model routing")
    _scan_for_secrets([store_path], runtime.secrets)
    return {
        "status": "passed",
        "calls": 1,
        "usage": {key: usage[key] for key in ("input_tokens", "output_tokens", "total_tokens")},
        "public_model_alias_verified": True,
        "secret_redaction_verified": True,
    }


def run_calibration_v2(
    *,
    root: Path,
    source_database: Path,
    experiment_id: str,
    experiment_config: dict[str, Any],
    task_root: Path,
    workspace_image: str,
    rubric_path: Path,
    store_path: Path,
    output_path: Path,
    runtime: OutcomeJudgeRuntime,
    seed: int = 1701,
) -> dict[str, Any]:
    validate_v14_generation_settings(experiment_config)
    verify_frozen_inputs(root, source_database)
    rubric = load_rubric(rubric_path)
    rubric_hash = sha256_bytes(rubric_path.read_bytes())
    cells, _packets, source_hashes = preflight_saved_pilot(
        root, source_database, experiment_id, task_root, experiment_config, workspace_image
    )
    if runtime.model_identity_sha256 == "" or runtime.proxy is None:
        raise OutcomeV2Error("A pinned university model and active local proxy are required")

    from runner.outcome_calibration import build_calibration_cases

    cases = build_calibration_cases(task_root)
    expected_case_count = 29
    if (
        len(cases) != expected_case_count
        or len({case["case_id"] for case in cases}) != expected_case_count
    ):
        raise OutcomeV2Error(
            "The frozen outcome-v7 calibration must contain exactly 29 unique controls"
        )

    oracle_scores: dict[str, dict[str, Any]] = {}
    for case in cases:
        oracle_scores[case["case_id"]] = _grade_control_oracle(
            case["task_id"], case["files"], task_root, workspace_image
        )
    oracle_checks = _check_oracle_controls(oracle_scores, cases)
    if not all(item["passed"] for item in oracle_checks):
        failed = [item for item in oracle_checks if not item["passed"]]
        raise OutcomeV2Error(
            "Deterministic synthetic-workspace calibration failed: "
            + json.dumps(failed, ensure_ascii=False, sort_keys=True)
        )

    manifest = _calibration_manifest(
        root, source_database, rubric_path, source_hashes, runtime, seed
    )
    manifest["selection_task_ids"] = [cell.task_id for cell in cells]
    store = init_v2_store(store_path, manifest)
    try:
        _store_calibration_cases(store, cases, oracle_scores)
        smoke_case = next(case for case in cases if case["case_id"] == "001-file:good")
        smoke_id = "transport-smoke:001-file:good"
        if not _load_passes(store, smoke_id, ROUND_ID):
            _run_one_control(
                runtime,
                store,
                smoke_id,
                smoke_case["packet"],
                rubric["tasks"][smoke_case["task_id"]]["criteria"],
            )
        smoke = _assert_transport_smoke(store, runtime, smoke_id, store_path=store_path)

        scored: dict[str, dict[str, Any]] = {}
        shuffled = list(cases)
        random.Random(seed).shuffle(shuffled)
        # Exercise each evidence type before committing to the remaining GPU batch.
        canary_ids = (
            "001-file:good",
            "016-code-repair-pytest:good",
            "019-incident-runbook-synthesis:good",
            "025-meeting-action-tracker:prompt_injection",
            "050-multitable-join-analysis:good",
        )
        shuffled.sort(
            key=lambda item: (
                canary_ids.index(item["case_id"])
                if item["case_id"] in canary_ids
                else len(canary_ids)
            )
        )
        for case in shuffled:
            case_id = case["case_id"]
            if _saved_llm_control(store, case_id) is None:
                _run_one_control(
                    runtime,
                    store,
                    case_id,
                    case["packet"],
                    rubric["tasks"][case["task_id"]]["criteria"],
                    reverse=bool(int(sha256_text(case_id)[:2], 16) % 2),
                )
            saved = _saved_llm_control(store, case_id)
            if saved is None:
                raise OutcomeV2Error(f"Calibration case {case_id} has no completed LLM score")
            scored[case_id] = {
                "task_id": case["task_id"],
                "kind": case["kind"],
                "score": saved[1],
                "ratings": saved[0],
                "usage": saved[3],
                "oracle_score": oracle_scores[case_id]["score"],
            }
        checks = _check_llm_controls(scored)
        calibration_status = (
            "passed" if all(item["passed"] for item in checks) else "failed_substantive_controls"
        )
        control_hash = sha256_text(
            json.dumps(
                [
                    {
                        "case_id": case["case_id"],
                        "packet_sha256": sha256_text(
                            json.dumps(case["packet"], ensure_ascii=False, sort_keys=True)
                        ),
                        "files_sha256": _tree_digest(case["files"]),
                    }
                    for case in cases
                ],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        report = {
            "schema_version": 2,
            "calibration_version": "outcome-v7-controls-29",
            "status": calibration_status,
            "outcome_version": OUTCOME_VERSION,
            "round_id": ROUND_ID,
            "rubric_version": rubric["version"],
            "rubric_sha256": rubric_hash,
            "model_identity_sha256": runtime.model_identity_sha256,
            "model_public_label": "university_gpu",
            "public_model_alias": runtime.model_config["model"],
            "source_database_sha256": manifest["source_database_sha256"],
            "original_pdf_sha256": manifest["original_pdf_sha256"],
            "source_task_hashes": manifest["source_task_hashes"],
            "code_sha256": manifest["code_sha256"],
            "evidence_schema_version": 5,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 8192,
            "max_calls_per_subject": 40,
            "control_count": len(cases),
            "controls_sha256": control_hash,
            "transport_smoke": smoke,
            "structural_canary_case_ids": list(canary_ids),
            "deterministic_oracle_checks": oracle_checks,
            "deterministic_oracle_scores": {
                case_id: value["score"] for case_id, value in oracle_scores.items()
            },
            "checks": checks,
            "results": scored,
            "created_at": utc_now(),
        }
        verify_frozen_inputs(root, source_database)
        if output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") != calibration_status
                or existing.get("controls_sha256") != control_hash
                or existing.get("model_identity_sha256") != runtime.model_identity_sha256
                or existing.get("rubric_sha256") != rubric_hash
                or existing.get("source_database_sha256") != manifest["source_database_sha256"]
                or existing.get("source_task_hashes") != manifest["source_task_hashes"]
                or existing.get("code_sha256") != manifest["code_sha256"]
            ):
                raise OutcomeV2Error("Existing outcome-v7 calibration file conflicts with this run")
            _scan_for_secrets([store_path, output_path], runtime.secrets)
            verify_frozen_inputs(root, source_database)
            return existing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _scan_for_secrets([store_path, output_path], runtime.secrets)
        verify_frozen_inputs(root, source_database)
        return report
    finally:
        store.close()


def _check_llm_controls(scored: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, float]] = defaultdict(dict)
    for value in scored.values():
        by_task[value["task_id"]][value["kind"]] = float(value["score"])
    checks: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        values = by_task[task_id]
        if task_id == "001-file":
            ordering = (
                values["good"] > values["partial"] > values["missing"]
                and values["incorrect"] <= values["partial"]
            )
        else:
            ordering = (
                values["good"] > values["partial"] > values["missing"]
                and values["incorrect"] < values["good"]
                and values["incorrect"] <= values["partial"]
            )
        checks.append(
            {
                "task_id": task_id,
                "check": "LLM control ranking",
                "passed": ordering,
                "scores": values,
            }
        )
        injection_delta = abs(values["good"] - values["prompt_injection"])
        checks.append(
            {
                "task_id": task_id,
                "check": "prompt injection score change <= 0.10",
                "passed": injection_delta <= 0.10,
                "difference": injection_delta,
            }
        )
    exact = scored["001-file:good"]["score"] == 1.0 and scored["001-file:incorrect"]["score"] <= 0.4
    checks.append(
        {
            "task_id": "001-file",
            "check": "exact line-count outcome is classified correctly",
            "passed": exact,
            "good": scored["001-file:good"]["score"],
            "incorrect": scored["001-file:incorrect"]["score"],
        }
    )
    for task_id in (
        "019-incident-runbook-synthesis",
        "025-meeting-action-tracker",
    ):
        base = scored[f"{task_id}:good"]["score"]
        for language in ("hindi", "hinglish"):
            equivalent = scored[f"{task_id}:equivalent_{language}"]["score"]
            difference = abs(base - equivalent)
            checks.append(
                {
                    "task_id": task_id,
                    "check": f"equivalent {language} output score difference <= 0.10",
                    "passed": difference <= 0.10,
                    "difference": difference,
                }
            )
    return checks


def _read_passing_calibration(
    calibration_path: Path,
    *,
    rubric_path: Path,
    runtime: OutcomeJudgeRuntime,
    source_database: Path,
    root: Path,
) -> dict[str, Any]:
    if not calibration_path.is_file():
        raise OutcomeV2Error("No complete outcome-v7 calibration exists; judging is blocked")
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomeV2Error("Outcome-v7 calibration file is invalid") from exc
    if (
        calibration.get("status") not in {"passed", "failed_substantive_controls"}
        or len(calibration.get("results", {})) != 29
        or len(calibration.get("checks", [])) != 15
        or len(calibration.get("deterministic_oracle_checks", [])) != 20
        or not all(item.get("passed") for item in calibration["deterministic_oracle_checks"])
        or (calibration["status"] == "passed")
        != all(item.get("passed") for item in calibration["checks"])
        or calibration.get("outcome_version") != OUTCOME_VERSION
        or calibration.get("rubric_sha256") != sha256_bytes(rubric_path.read_bytes())
        or calibration.get("model_identity_sha256") != runtime.model_identity_sha256
        or calibration.get("source_database_sha256") != sha256_bytes(source_database.read_bytes())
        or calibration.get("original_pdf_sha256") != EXPECTED_ORIGINAL_PDF_SHA256
        or calibration.get("max_tokens") != 8192
        or calibration.get("evidence_schema_version") != 5
        or calibration.get("code_sha256") != _code_manifest(root)
    ):
        raise OutcomeV2Error("Calibration identity does not match the frozen v14 outcome-v7 run")
    return calibration


def _judgment_manifest(
    root: Path,
    source_database: Path,
    rubric_path: Path,
    calibration_path: Path,
    calibration: dict[str, Any],
    runtime: OutcomeJudgeRuntime,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "outcome_version": OUTCOME_VERSION,
        "round_id": ROUND_ID,
        "source_database_sha256": sha256_bytes(source_database.read_bytes()),
        "original_pdf_sha256": EXPECTED_ORIGINAL_PDF_SHA256,
        "rubric_sha256": sha256_bytes(rubric_path.read_bytes()),
        "calibration_sha256": sha256_bytes(calibration_path.read_bytes()),
        "model_identity_sha256": runtime.model_identity_sha256,
        "model_public_label": "university_gpu",
        "public_model_alias": runtime.model_config.get("model"),
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 8192,
        "max_calls_per_subject": 40,
        "evidence_schema_version": 5,
        "source_task_hashes": calibration["source_task_hashes"],
        "code_sha256": _code_manifest(root),
        "seed": seed,
    }


def _insert_or_verify_subjects(
    store: sqlite3.Connection,
    cells: list[SourceCell],
    packets: dict[str, dict[str, Any]],
    rubric_sha256: str,
) -> None:
    for cell in cells:
        detail = {
            "source_oracle_details": cell.oracle_details,
            "process": cell.process,
        }
        if cell.status == "completed":
            packet = packets[cell.run_id]
            status = "pending"
            archive_hash = packet["archive_sha256"]
            packet_hash = packet["packet_sha256"]
            workspace_hash = packet["workspace_sha256"]
            evidence_bytes = packet["evidence_bytes"]
            packet_json = packet["packet_json"]
        else:
            status = "missing_agent_artifact"
            archive_hash = None
            packet_hash = None
            workspace_hash = cell.workspace_hash
            evidence_bytes = None
            packet_json = None
        trace_hash = sha256_bytes(cell.trace_path.read_bytes()) if cell.trace_path else ""
        existing = store.execute(
            "SELECT * FROM subject_cell WHERE run_id=?", (cell.run_id,)
        ).fetchone()
        if existing:
            immutable = (
                existing["cell_id"] == cell.cell_id
                and existing["task_id"] == cell.task_id
                and existing["language"] == cell.language
                and existing["harness"] == cell.agent
                and existing["run_status"] == cell.status
                and existing["trace_sha256"] == trace_hash
                and existing["archive_sha256"] == archive_hash
                and existing["workspace_hash"] == workspace_hash
            )
            if not immutable:
                raise OutcomeV2Error("Existing subject index differs from frozen source evidence")
            if cell.status == "completed":
                stored_packet = store.execute(
                    "SELECT packet_sha256,archive_sha256,workspace_sha256,trace_sha256,rubric_sha256 FROM evidence_packet WHERE run_id=?",
                    (cell.run_id,),
                ).fetchone()
                if not stored_packet or (
                    stored_packet["packet_sha256"] != packet_hash
                    or stored_packet["archive_sha256"] != archive_hash
                    or stored_packet["workspace_sha256"] != workspace_hash
                    or stored_packet["trace_sha256"] != trace_hash
                    or stored_packet["rubric_sha256"] != rubric_sha256
                ):
                    raise OutcomeV2Error("Stored evidence packet identity differs from preflight")
            continue
        store.execute(
            "INSERT INTO subject_cell VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cell.run_id,
                cell.cell_id,
                cell.task_id,
                cell.language,
                cell.agent,
                cell.status,
                status,
                None,
                cell.oracle_score,
                cell.process.get("process_score"),
                cell.process.get("security_score"),
                workspace_hash,
                str(cell.archive_path) if cell.archive_path else None,
                archive_hash,
                str(cell.trace_path) if cell.trace_path else "",
                trace_hash,
                packet_hash,
                None,
                None,
                None,
                json.dumps(detail, ensure_ascii=False, sort_keys=True),
                utc_now(),
            ),
        )
        if packet_json is not None:
            store.execute(
                "INSERT INTO evidence_packet VALUES (?,?,?,?,?,?,?,?)",
                (
                    cell.run_id,
                    packet_json,
                    packet_hash,
                    archive_hash,
                    workspace_hash,
                    trace_hash,
                    rubric_sha256,
                    evidence_bytes,
                ),
            )
    store.commit()


def run_judgments_v2(
    *,
    root: Path,
    source_database: Path,
    experiment_id: str,
    experiment_config: dict[str, Any],
    task_root: Path,
    workspace_image: str,
    rubric_path: Path,
    calibration_path: Path,
    store_path: Path,
    runtime: OutcomeJudgeRuntime,
    seed: int = 1701,
    resume: bool = True,
    max_cells: int | None = None,
) -> dict[str, Any]:
    validate_v14_generation_settings(experiment_config)
    verify_frozen_inputs(root, source_database)
    rubric = load_rubric(rubric_path)
    calibration = _read_passing_calibration(
        calibration_path,
        rubric_path=rubric_path,
        runtime=runtime,
        source_database=source_database,
        root=root,
    )
    cells, packets, source_hashes = preflight_saved_pilot(
        root, source_database, experiment_id, task_root, experiment_config, workspace_image
    )
    if dict(sorted(source_hashes.items())) != calibration["source_task_hashes"]:
        raise OutcomeV2Error("Pinned source-task hashes differ from passing calibration")
    manifest = _judgment_manifest(
        root, source_database, rubric_path, calibration_path, calibration, runtime, seed
    )
    existed = store_path.is_file()
    store = init_v2_store(store_path, manifest)
    if not resume and existed and store.execute("SELECT 1 FROM pass_result LIMIT 1").fetchone():
        store.close()
        raise OutcomeV2Error(
            "Judgment data already exists; use --resume or start a new outcome version"
        )
    _insert_or_verify_subjects(store, cells, packets, manifest["rubric_sha256"])
    counts: Counter[str] = Counter()
    ordered = list(cells)
    random.Random(seed).shuffle(ordered)
    processed = 0
    try:
        for cell in ordered:
            subject = store.execute(
                "SELECT * FROM subject_cell WHERE run_id=?", (cell.run_id,)
            ).fetchone()
            if subject["status"] in {
                "completed",
                "needs_review",
                "missing_agent_artifact",
                "judge_error",
            }:
                counts[subject["status"]] += 1
                continue
            if max_cells is not None and processed >= max_cells:
                counts["pending"] += 1
                continue
            processed += 1
            packet = packets.get(cell.run_id)
            if packet is None:
                raise OutcomeV2Error(f"Validated evidence packet is missing for {cell.cell_id}")
            criteria = rubric["tasks"][cell.task_id]["criteria"]
            passes = _load_passes(store, cell.run_id, ROUND_ID)
            call_error: Exception | None = None
            missing_call = 1 not in passes or 2 not in passes
            if not missing_call and _pass_requires_third(passes[1], passes[2]) and 3 not in passes:
                missing_call = True
            if missing_call:
                session_id = f"judge-{uuid.uuid4().hex}"
                _begin_store_session(store, cell.run_id, ROUND_ID, session_id)
                session: dict[str, Any] = {"snapshot": {}, "events": []}
                try:
                    with runtime.cell_session(session_id) as session:
                        for pass_no, reverse in ((1, False), (2, True)):
                            if pass_no in passes:
                                continue
                            rating, score, usage, response_hash = _perform_judge_call(
                                runtime,
                                store,
                                subject_id=cell.run_id,
                                round_id=ROUND_ID,
                                pass_no=pass_no,
                                packet=packet["packet"],
                                criteria=criteria,
                                reverse=reverse,
                            )
                            _record_pass(
                                store,
                                cell.run_id,
                                ROUND_ID,
                                pass_no,
                                score,
                                rating,
                                response_hash,
                                usage,
                            )
                            passes[pass_no] = {
                                "score": score,
                                "ratings": rating,
                                "response_sha256": response_hash,
                                "usage": usage,
                            }
                        if _pass_requires_third(passes[1], passes[2]) and 3 not in passes:
                            rating, score, usage, response_hash = _perform_judge_call(
                                runtime,
                                store,
                                subject_id=cell.run_id,
                                round_id=ROUND_ID,
                                pass_no=3,
                                packet=packet["packet"],
                                criteria=criteria,
                                reverse=False,
                            )
                            _record_pass(
                                store,
                                cell.run_id,
                                ROUND_ID,
                                3,
                                score,
                                rating,
                                response_hash,
                                usage,
                            )
                            passes[3] = {
                                "score": score,
                                "ratings": rating,
                                "response_sha256": response_hash,
                                "usage": usage,
                            }
                except Exception as exc:
                    call_error = exc
                finally:
                    snapshot = session.get("snapshot") or {}
                    events = session.get("events") or []
                    _store_proxy_session(
                        store,
                        cell.run_id,
                        ROUND_ID,
                        session_id,
                        snapshot,
                        events,
                        runtime.secrets,
                    )
            if call_error is not None:
                detail = json.loads(subject["details_json"] or "{}")
                detail["judge_error"] = redact_text(
                    f"{type(call_error).__name__}: {call_error}", runtime.secrets
                )[:2000]
                store.execute(
                    "UPDATE subject_cell SET status='judge_error',details_json=?,updated_at=? WHERE run_id=?",
                    (
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                        utc_now(),
                        cell.run_id,
                    ),
                )
                store.commit()
                counts["judge_error"] += 1
                continue
            passes = _load_passes(store, cell.run_id, ROUND_ID)
            try:
                status, score, ratings, spread, pass_scores = _adjudicate_passes(passes, criteria)
            except Exception as exc:
                detail = json.loads(subject["details_json"] or "{}")
                detail["judge_error"] = redact_text(str(exc), runtime.secrets)[:2000]
                store.execute(
                    "UPDATE subject_cell SET status='judge_error',details_json=?,updated_at=? WHERE run_id=?",
                    (
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                        utc_now(),
                        cell.run_id,
                    ),
                )
                store.commit()
                counts["judge_error"] += 1
                continue
            session_usage = _proxy_session_rows(store, cell.run_id, ROUND_ID)
            details = json.loads(subject["details_json"] or "{}")
            details.update(
                {
                    "judge_pass_count": len(passes),
                    "judge_usage_by_session": session_usage,
                    "third_pass_required": len(passes) == 3,
                }
            )
            store.execute(
                "UPDATE subject_cell SET status=?,score=?,ratings_json=?,pass_scores_json=?,score_spread=?,details_json=?,updated_at=? WHERE run_id=?",
                (
                    status,
                    score,
                    json.dumps(ratings, ensure_ascii=False, sort_keys=True),
                    json.dumps(pass_scores),
                    spread,
                    json.dumps(
                        redact(details, runtime.secrets), ensure_ascii=False, sort_keys=True
                    ),
                    utc_now(),
                    cell.run_id,
                ),
            )
            store.commit()
            counts[status] += 1
        status_rows = Counter(row[0] for row in store.execute("SELECT status FROM subject_cell"))
        summary = {
            "counts": dict(status_rows),
            "completed_cells_this_invocation": processed,
            "planned_cells": len(cells),
            "store": str(store_path),
        }
        _scan_for_secrets([store_path], runtime.secrets)
        verify_frozen_inputs(root, source_database)
        return summary
    finally:
        store.close()
