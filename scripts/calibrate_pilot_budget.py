"""Select a generation cap on synthetic tool calls before pilot cells exist."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from runner.config import ExperimentConfig
from runner.inference import ensure_model_manifest
from runner.proxy import InferenceProxy


PROBES = (
    "Call write_file with path /workspace/out/check.json and a JSON object containing ids A, B, C and their sum 6.",
    "Call write_file with path /workspace/out/ledger.json and a JSON array of records "
    "for integers 1 through 180. Each record must contain integer id, square, and parity fields. "
    "Put the full valid JSON array in the tool's content argument in one call.",
)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a UTF-8 file in the synthetic calibration workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.corrected.pilot-v13.yaml"))
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    output = config.root / config.inference["calibration_manifest"]
    if output.exists():
        raise FileExistsError("Calibration is frozen; create a new experiment version to change it")
    endpoint, _ = ensure_model_manifest(config.root, config.inference, verify_tool=True)
    trials: list[dict[str, object]] = []
    selected = None
    for cap in (4096, 8192):
        proxy = InferenceProxy(
            endpoint.base_url, endpoint.api_key, endpoint.resolved_model,
            public_model="phase1-university-model", max_calls_per_cell=40,
            temperature=0.0, top_p=1.0, max_tokens=cap,
        ).start()
        client = OpenAI(base_url=proxy.base_url, api_key=proxy.client_key, timeout=90)
        passed = True
        try:
            proxy.begin_cell(f"synthetic-calibration-{cap}")
            for index, prompt in enumerate(PROBES):
                try:
                    response = client.chat.completions.create(
                        model=proxy.public_model,
                        messages=[{"role": "system", "content": "Return a valid tool call without private reasoning."},
                                  {"role": "user", "content": prompt}],
                        tools=TOOLS, tool_choice={"type": "function", "function": {"name": "write_file"}},
                        temperature=0, top_p=1, max_tokens=cap,
                    )
                    choice = response.choices[0]
                    calls = choice.message.tool_calls or []
                    valid = choice.finish_reason != "length" and len(calls) == 1
                    if valid:
                        args_value = json.loads(calls[0].function.arguments)
                        expected_path = "/workspace/out/check.json" if index == 0 else "/workspace/out/ledger.json"
                        valid = calls[0].function.name == "write_file" and args_value.get("path") == expected_path
                        if valid:
                            content = json.loads(args_value.get("content", ""))
                            if index == 0:
                                valid = isinstance(content, dict) and content.get("sum") == 6
                            else:
                                valid = (isinstance(content, list) and len(content) == 180
                                         and [item.get("id") for item in content] == list(range(1, 181))
                                         and all(item.get("square") == item["id"] ** 2
                                                 and item.get("parity") == ("even" if item["id"] % 2 == 0 else "odd")
                                                 for item in content))
                    trials.append({"cap": cap, "probe": index, "valid_tool_call": valid,
                                   "finish_reason": choice.finish_reason,
                                   "completion_tokens": response.usage.completion_tokens if response.usage else None})
                    passed = passed and valid
                except Exception as exc:
                    trials.append({"cap": cap, "probe": index, "valid_tool_call": False,
                                   "error_type": type(exc).__name__})
                    passed = False
            proxy.end_cell()
        finally:
            client.close()
            proxy.close()
        if passed:
            selected = cap
            break
    if selected is None:
        raise RuntimeError("Neither synthetic calibration cap produced valid tool calls; pilot remains blocked")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_max_tokens": selected,
        "served_model_sha256": hashlib.sha256(endpoint.resolved_model.encode()).hexdigest(),
        "probes_sha256": hashlib.sha256(json.dumps(PROBES).encode()).hexdigest(),
        "trials": trials,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Synthetic calibration selected {selected} tokens per call; freeze the same value in the v13 config")


if __name__ == "__main__":
    main()
