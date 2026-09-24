# Phase I pilot outcome rejudgment v3

## Version boundary

Outcome-v2 stopped at offline deterministic preflight. It made no university
LLM calls, produced no calibration report, and produced no pilot outcome
scores. The v2 SQLite store and calibration JSON were not created. The
preflight exposed a bad task-016 missing-artifact control, so all LLM
judgments use the new version and separate ignored directory
`data/phase1/corrected/pilot-v14/rejudgments/outcome-v3/`.

The v14 run database remains pinned at SHA-256
`d62e4753eb45fec807427e6866d88a34966b78cfa2fc1815689d2e497cccfecb` and the
original v14 PDF at
`bd5b26f3f93bba48167e8c339a9656e8232c61fd323d0d03ecc6b08e9ffd2780`. The
implementation verifies both hashes before and after preflight, calibration,
judging, analysis, and PDF creation. It never constructs `ExperimentRunner`
or `RunStore` from outcome commands. V1 failed-start records remain an audit
trail; the 216-run study, v14 grades, and v14 report are not rewritten.

## Frozen rubric and input evidence

`configs/outcome-rubric.pilot-v14-v3.yaml` keeps the approved criterion
weights. Each criterion receives a 0-4 ordinal rating; software computes the
weighted result in [0,1]. The judge receives canonical English requirements,
reference facts, task inputs/outputs, and independent task-016 pytest evidence.
Every citation identifies a stable evidence ID and matching path, plus a
validated line, JSON pointer, test observation, or exact short quote. Evidence
is hash-checked and never silently truncated.

Before any LLM request, preflight checks all 24 imported task source hashes,
the exact 45-cell v14 matrix, 44 workspace archives, 45 traces, and all nine
archived task-016 pytest runs. It also grades all 29 synthetic workspaces with
the pinned upstream task oracles in isolated Docker containers.

## Task-016 oracle floor adjustment

The v2 task-016 `missing` control accidentally reused the corrected
agent-produced `config_manager.py`; it therefore scored 0.90. V3 defines
`missing` from untouched upstream fixtures only. The upstream task-016 oracle
then returns 0.27 because it awards 0.30 for an unchanged test-file hash and
multiplies by 0.90 when the original broken implementation violates its code
constraint. Thus, its task-specific missing-score ceiling is 0.30, while the
other four tasks retain the frozen 0.25 ceiling. The partial control must
still score between correct and missing. Removing the test fixture merely to
force the score below 0.25 would corrupt the input evidence, so v3 preserves
the upstream workspace contract and records this exception.

Task 001 has a different upstream limitation: its oracle is binary and checks
only whether the output is exactly `4`. Its partial control retains the
correct count but modifies the input, demonstrating that the oracle ignores
input preservation. The LLM rubric must score this partial control below the
good control and above missing.

## Runtime, retries, and resuming

The dedicated runtime re-queries the university model list and matches the
exact identity already pinned by v14. It uses a local proxy with temperature
0, top-p 1, an 8,192-token request ceiling, and at most 40 calls per subject.
Only the public model alias appears in client responses and stored events.
Credentials, private endpoint, and private model ID are held in memory and
scanned against generated records.

One live task-001 correct-control request validates JSON, usage, routing, and
redaction before the other 29 LLM calibration controls. Calibration is
resumable by case. A passing `calibration.json` is emitted only after the
deterministic and LLM gates all pass.

The 44 eligible saved runs are shuffled with seed 1701 and receive two ratings
with reversed criterion order. A third rating is requested for a weighted
score difference over 0.15 or a criterion difference of at least two levels.
Three ratings use the per-criterion median. A three-score range over 0.20
marks the result `needs_review` with no primary score. Malformed responses and
transient connection, timeout, 429, and 5xx errors receive at most two
retries; deterministic 4xx policy errors do not retry. SQLite call attempts,
proxy events, response hashes, redacted responses, citations, errors, and
usage are retained append-only. Resume never duplicates a completed pass.

## Analysis and release

The report requires all 45 cells and traces, 44 valid two- or three-pass
judgments, exactly one known missing agent artifact, passing calibration, and
matching source, task, rubric, evidence, model, and code hashes. It shows the
LLM outcome as primary, the original oracle, process score, and security gate
separately, plus an LLM-outcome x process x security diagnostic. Missing
scores and missing token usage stay missing.

Language contrasts use task-balanced paired Hindi-English and
Hinglish-English differences with 20,000 task-cluster bootstrap draws (seed
1701). The local PDF includes the calibration, coverage, task and condition
tables, task heatmap, paired plots, score distributions, LLM-oracle comparison,
criterion ratings, and clickable trace/workspace links. A blinded 12-cell
review packet has exactly four examples per language and harness, at least
two per task, and low/middle/high disagreement coverage. Reviewer forms are
independent; the unblinding key is a separate ignored file.

All results remain retrospective and exploratory. There are five purposively
selected tasks, one execution per condition, one outcome-judge model family
also used for agents, and no independent bilingual review yet. This study is
not directly numerically comparable with the original Harness-Bench
leaderboard.

```powershell
py -3.14 -m runner.cli outcome-preflight --config configs/phase1.corrected.pilot-v14.yaml
py -3.14 -m runner.cli calibrate-outcomes --config configs/phase1.corrected.pilot-v14.yaml
py -3.14 -m runner.cli judge-outcomes --config configs/phase1.corrected.pilot-v14.yaml --resume
py -3.14 -m runner.cli outcome-report --config configs/phase1.corrected.pilot-v14.yaml
```
