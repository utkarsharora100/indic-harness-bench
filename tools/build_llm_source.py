#!/usr/bin/env python3
"""Build a single Markdown corpus with every text file from a pinned Git commit."""
from __future__ import annotations
import hashlib
import pathlib
import subprocess

REPO_COMMIT = "f16a2233a2769f06598c02960845135ba5b3e5d8"
OUTPUT = pathlib.Path("LLM_SOURCE.md")
BINARY_HINTS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".zip", ".gz", ".sqlite", ".db", ".woff", ".woff2", ".ttf", ".otf"}
tick = chr(96)

def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])

def lang_for(path: str) -> str:
    p = pathlib.PurePosixPath(path)
    ext = p.suffix.lower()
    if ext == ".py": return "python"
    if ext in {".yaml", ".yml"}: return "yaml"
    if ext == ".json": return "json"
    if ext == ".toml": return "toml"
    if ext == ".md": return "markdown"
    if ext in {".sh", ".bash"}: return "bash"
    if ext == ".sql": return "sql"
    if ext == ".html": return "html"
    if ext == ".css": return "css"
    if ext in {".js", ".jsx"}: return "javascript"
    if ext in {".ts", ".tsx"}: return "typescript"
    if ext in {".xml", ".svg"}: return "xml"
    if ext == ".csv": return "csv"
    if p.name.lower() == "dockerfile" or p.name.lower().startswith("dockerfile."): return "dockerfile"
    if p.name == ".gitignore": return "gitignore"
    return "text"

def fence_for(body: str) -> str:
    longest = run = 0
    for char in body:
        if char == tick:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return tick * max(3, longest + 1)

paths = [p.decode("utf-8") for p in git_bytes("ls-tree", "-r", "-z", "--name-only", REPO_COMMIT).split(b"\0") if p]
entries = []
for path in paths:
    raw = git_bytes("show", f"{REPO_COMMIT}:{path}")
    suffix = pathlib.PurePosixPath(path).suffix.lower()
    try:
        decoded = raw.decode("utf-8")
        if "\0" in decoded or suffix in BINARY_HINTS:
            entries.append((path, raw, None))
        else:
            entries.append((path, raw, decoded))
    except UnicodeDecodeError:
        entries.append((path, raw, None))

text_entries = [(p, raw, body) for p, raw, body in entries if body is not None]
binary_entries = [(p, raw) for p, raw, body in entries if body is None]

def anchor(path: str) -> str:
    return "".join(ch for ch in path.lower() if ch.isalnum())

out = [
    "# Indic-Harness-Bench — complete LLM-friendly source corpus", "",
    "Every UTF-8 text file from the pinned Git commit is embedded in full below under its exact repository path. File bodies are not replaced with links or summaries. Binary assets are catalogued separately.", "",
    "## Snapshot", "",
    "- Repository: " + tick + "utkarsharora100/indic-harness-bench" + tick,
    "- Original branch: " + tick + "prishiv_dev" + tick,
    "- Pinned source commit: " + tick + REPO_COMMIT + tick,
    f"- Tracked files: {len(paths)}",
    f"- Full text files embedded: {len(text_entries)}",
    f"- Binary files catalogued: {len(binary_entries)}", "",
    "## Contents", ""
]
for path, _, _ in text_entries:
    out.append(f"- [{path}](#{anchor(path)})")
out.extend(["", "---", ""])
for path, raw, body in text_entries:
    fence = fence_for(body)
    out.extend([
        f'<a id="{anchor(path)}"></a>',
        "## FILE: " + tick + path + tick, "",
        "- Original size: " + str(len(raw)) + " bytes",
        "- Git blob SHA: " + tick + git_bytes("rev-parse", f"{REPO_COMMIT}:{path}").decode().strip() + tick, "",
        fence + lang_for(path),
        body.rstrip("\n"),
        fence, "", "---", ""
    ])
out.extend(["## Binary files (not embedded as text)", ""])
for path, raw in binary_entries:
    out.extend([
        "### " + tick + path + tick, "",
        f"- Size: {len(raw)} bytes",
        "- SHA-256: " + tick + hashlib.sha256(raw).hexdigest() + tick, ""
    ])
out.extend([
    "## Verification", "",
    f"- Source tree file count: {len(paths)}",
    f"- Full text bodies embedded: {len(text_entries)}",
    f"- Binary files catalogued: {len(binary_entries)}",
    "- Content was read directly from the pinned Git commit.", ""
])
OUTPUT.write_text("\n".join(out), encoding="utf-8", newline="\n")
print(f"Created {OUTPUT}: {OUTPUT.stat().st_size} bytes; {len(paths)} tracked files; {len(text_entries)} full text bodies.")
