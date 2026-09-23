"""Mechanical v13 manifest migration; instruction text is never regenerated."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from benchmark.canonical import blob_tree_sha256, task_blobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("benchmark/task_selection.yaml"))
    parser.add_argument("--translations", type=Path, default=Path("benchmark/translations/phase1.yaml"))
    parser.add_argument("--selection-output", type=Path, default=Path("benchmark/task_selection.v13.yaml"))
    parser.add_argument("--translations-output", type=Path, default=Path("benchmark/translations/phase1.v13.yaml"))
    args = parser.parse_args()
    if args.selection_output.exists() or args.translations_output.exists():
        raise FileExistsError("v13 manifests already exist; they are immutable")
    selection = yaml.safe_load(args.selection.read_text(encoding="utf-8"))
    translations = yaml.safe_load(args.translations.read_text(encoding="utf-8"))
    revision = selection["source"]["revision"]
    replacements: dict[str, str] = {}
    for item in selection["tasks"]:
        task_id = item["task_id"]
        old = item["source_sha256"]
        if old != translations["tasks"][task_id]["source_sha256"]:
            raise ValueError(f"Old source hash differs in translation metadata: {task_id}")
        blobs = task_blobs(args.source, revision, task_id)
        prompt = blobs["prompt.txt"][0].decode("utf-8").replace("\r\n", "\n")
        if prompt != translations["tasks"][task_id]["instructions"]["english"]:
            raise ValueError(f"English canonical prompt changed: {task_id}")
        replacements[old] = blob_tree_sha256(blobs)
    if len(replacements) != len(selection["tasks"]):
        raise ValueError("Source hash migration is ambiguous")
    for origin, target in ((args.selection, args.selection_output),
                           (args.translations, args.translations_output)):
        content = origin.read_text(encoding="utf-8")
        for old, new in replacements.items():
            content = content.replace(old, new)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "# v13: hashes refer to raw blobs at the pinned Git commit; prompt text is unchanged.\n"
            + content,
            encoding="utf-8",
            newline="\n",
        )
    print(f"Created immutable canonical manifests for {len(replacements)} tasks")


if __name__ == "__main__":
    main()
