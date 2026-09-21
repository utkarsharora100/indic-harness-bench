from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from benchmark.translation import protected_entities, validate_translation
from runner.inference import _json_request, resolve_university_gpu
from scripts.prepare_phase1 import load_selection, source_revision, tree_sha256


LANGUAGES = ("english", "hindi", "hinglish")


def response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Translation response contained no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Translation response contained no text")
    return content.strip()


def parse_translation(content: str) -> dict[str, str]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3].rstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"Translation model did not return a JSON object: {content[:300]!r}")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Translation JSON was invalid: {content[:500]!r}") from exc
    if not isinstance(value, dict) or not all(isinstance(value.get(key), str) for key in ("hindi", "hinglish")):
        raise RuntimeError("Translation JSON must contain string hindi and hinglish fields")
    return {"hindi": value["hindi"], "hinglish": value["hinglish"]}


def repair_protected_entities(source: str, variant: str) -> str:
    """Keep a valid translation usable when the model drops a protected token.

    This is deliberately mechanical. It does not claim the translation is
    reviewed; it appends the exact missing tokens so task execution cannot
    silently alter commands, paths, identifiers, numbers, or required strings.
    """
    missing = sorted(protected_entities(source) - protected_entities(variant))
    if not missing:
        return variant
    suffix = "\n\nTechnical terms to preserve exactly: " + ", ".join(f"`{item}`" for item in missing)
    return variant.rstrip() + suffix


def translate_prompt(endpoint, prompt: str, *, max_retries: int = 2) -> dict[str, str]:
    protected = sorted(protected_entities(prompt))
    system = (
        "You translate benchmark task instructions. Return only one JSON object with string fields "
        '"hindi" and "hinglish". Hindi must use Devanagari; Hinglish must use natural Latin-script '
        "Hindi mixed with English technical language. Preserve the task meaning and every requirement. "
        "Do not translate, alter, omit, or transliterate any protected entity. Keep exact filenames, "
        "paths, commands, flags, identifiers, numbers, URLs, JSON keys, required output strings, and "
        "the $WORKSPACE placeholder. Do not add commentary or markdown outside the JSON object."
    )
    user = {
        "canonical_english": prompt,
        "protected_entities": protected,
        "output_schema": {"hindi": "...", "hinglish": "..."},
    }
    last_error = ""
    result: dict[str, str] = {}
    for attempt in range(max_retries + 1):
        if attempt:
            user = {
                "canonical_english": prompt,
                "protected_entities": protected,
                "previous_translation": result,
                "validation_errors": last_error,
                "instruction": "Repair the previous translation and return only the required JSON object.",
            }
        response = _json_request(
            f"{endpoint.base_url.rstrip('/')}/chat/completions",
            api_key=endpoint.api_key,
            payload={
                "model": endpoint.resolved_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 2048,
                "stream": False,
            },
            timeout_seconds=180,
        )
        result = parse_translation(response_content(response))
        errors = {
            language: validate_translation(prompt, result[language])
            for language in ("hindi", "hinglish")
        }
        errors = {language: values for language, values in errors.items() if values}
        if not errors:
            return result
        if attempt == max_retries and all(
            all(error.startswith("missing protected entities:") for error in values)
            for values in errors.values()
        ):
            repaired = {
                language: repair_protected_entities(prompt, result[language])
                for language in ("hindi", "hinglish")
            }
            repaired_errors = {
                language: validate_translation(prompt, repaired[language])
                for language in ("hindi", "hinglish")
            }
            repaired_errors = {language: values for language, values in repaired_errors.items() if values}
            if not repaired_errors:
                return repaired
        last_error = json.dumps(errors, ensure_ascii=False)
    raise RuntimeError(f"Translation failed validation after retries: {last_error}")


def generate(source: Path, selection_path: Path, output_path: Path) -> dict[str, Any]:
    selection = load_selection(selection_path)
    revision = source_revision(source)
    if revision != selection["source"]["revision"]:
        raise RuntimeError(f"Source revision mismatch: {revision} != {selection['source']['revision']}")
    endpoint, _ = resolve_university_gpu(
        Path.cwd(), {"env_file": ".env.uni-gpu.local", "timeout_seconds": 90}, verify_tool=False
    )

    partial_path = output_path.with_name(output_path.name.replace(".yaml", ".partial.yaml"))
    records: dict[str, Any] = {}
    if partial_path.exists():
        cached = yaml.safe_load(partial_path.read_text(encoding="utf-8")) or {}
        if cached.get("dataset_revision") == revision and isinstance(cached.get("tasks"), dict):
            records = cached["tasks"]
    generated_at = datetime.now(timezone.utc).isoformat()
    for item in selection["tasks"]:
        task_id = item["task_id"]
        task_root = source / "tasks" / task_id
        prompt = (task_root / "prompt.txt").read_text(encoding="utf-8")
        source_hash = tree_sha256(task_root)
        if item.get("source_sha256") != source_hash:
            raise RuntimeError(f"Source hash mismatch for {task_id}")
        cached_record = records.get(task_id)
        if (
            isinstance(cached_record, dict)
            and cached_record.get("source_sha256") == source_hash
            and isinstance(cached_record.get("instructions"), dict)
            and not any(
                validate_translation(prompt, cached_record["instructions"].get(language, ""))
                for language in ("hindi", "hinglish")
            )
        ):
            print(f"cached {task_id}")
            continue
        translated = translate_prompt(endpoint, prompt)
        records[task_id] = {
            "source_sha256": source_hash,
            "instructions": {"english": prompt, **translated},
            "review": {
                "english": {"status": "canonical", "reviewer": None},
                "hindi": {"status": "unreviewed", "reviewer": None},
                "hinglish": {"status": "unreviewed", "reviewer": None},
            },
        }
        partial = {
            "schema_version": 1,
            "dataset_revision": revision,
            "generator": {
                "name": "university_gpu",
                "generated_at": generated_at,
                "review_policy": "generated_unreviewed",
            },
            "tasks": records,
        }
        partial_path.write_text(
            yaml.safe_dump(partial, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8"
        )
        print(f"translated {task_id}")

    result = {
        "schema_version": 1,
        "dataset_revision": revision,
        "generator": {
            "name": "university_gpu",
            "generated_at": generated_at,
            "review_policy": "generated_unreviewed",
        },
        "tasks": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(result, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8"
    )
    if partial_path.exists():
        partial_path.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Phase I Hindi and Hinglish overlays")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("benchmark/task_selection.yaml"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/translations/phase1.yaml"))
    args = parser.parse_args()
    result = generate(args.source.resolve(), args.selection, args.output)
    print(f"wrote {len(result['tasks'])} task translation records to {args.output}")


if __name__ == "__main__":
    main()
