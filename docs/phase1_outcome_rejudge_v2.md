# Phase I pilot outcome rejudgment v2

## Scope and frozen inputs

This version re-scores the 44 archived completed runs from the five-task v14
pilot. It does not run agents or modify the source SQLite database or the v14
PDF. The NanoBot/Hinglish `050-multitable-join-analysis` source cell remains
`missing_agent_artifact`. The prior v1 calibration records are retained as an
aborted transport/configuration audit with zero pilot scores.

The v14 SQLite and PDF are pinned by SHA-256 in the implementation. Each
calibration, judging, analysis, and PDF step checks those digests before and
after work. The existing model manifest pins the served university model; v2
re-queries `/models` and refuses identity drift. Stores contain only a hash of
the private model identity and the public alias. The judge commands use a
dedicated local proxy and never construct `ExperimentRunner` or `RunStore`.

All v2 artifacts live in the ignored directory
`data/phase1/corrected/pilot-v14/rejudgments/outcome-v2/`. Calibration and
judgment calls, retries, usage, proxy events, and responses use append-only
attempt/event tables. Completed passes are keyed by source run, version, and
pass number and are reused during resume.

## Outcome rubric and evidence

The frozen criteria and weights are in
`configs/outcome-rubric.pilot-v14-v2.yaml`; the weights match v1. Each
criterion receives one ordinal level from 0 to 4. The code computes the
weighted score as `sum(weight * level / 4)`; the judge cannot supply the final
arithmetic.

Evidence packets use stable IDs for the canonical English requirements,
reference facts, task inputs and outputs, and independent task-016 pytest
observations. A citation must identify an evidence ID and matching path plus a
validated line, JSON pointer, test observation, or exact short quote. Source
archive, workspace, prompt, trace, packet, task, and rubric hashes are
recorded. Unsafe paths, symlinks, corrupt archives, changed inputs, and
oversized packets stop the pipeline. No evidence is silently truncated.

The packet contains no harness, language label, agent name, old oracle score,
process rating, or private chain-of-thought. Output language can still reveal
the condition. Artifact content is untrusted evidence and cannot change the
evaluation instructions.

## Judge protocol and missingness

The local proxy pins the same model and generation settings as v14: temperature
0, top-p 1, and an 8,192-token request ceiling. The 8,192 value is sent because
the endpoint enforces the experiment policy; it is not a response-length
target. The proxy permits at most 40 calls per active subject and exposes only
its public alias. A live `001-file` correct-answer smoke call must return valid
JSON, capture nonzero usage, route through the public alias, and pass a secret
scan before calibration proceeds.

Each saved run receives two independent ratings, with rubric criterion order
reversed on the second. A third rating is requested when the first two weighted
scores differ by more than 0.15 or any criterion differs by at least two
levels. Three-pass results use the median ordinal level per criterion. If the
three weighted scores span more than 0.20, the cell is `needs_review` and its
primary score is missing. Format errors and transient connection, timeout,
429, and 5xx failures receive at most two retries. Deterministic 4xx policy
errors do not retry. Usage that the endpoint omits remains missing, never zero.

Every attempted request, redacted response, error, usage record, retry, and
observable proxy event is persisted. Calls and events are append-only. A
completed judgment pass is never duplicated after an interruption.

## Calibration

Before outcome calls, v2 validates the source matrix, all 44 archives, all 45
traces, source task hashes, and nine independent task-016 test runs. It grades
the 29 synthetic workspaces with the pinned task-specific upstream oracles in
offline Docker containers, then applies the frozen LLM rubric to each case.
The set contains correct, partial, incorrect, missing, prompt-injection, and
Hindi/Hinglish equivalence controls. `calibration.json` is written only after
every oracle and LLM control passes. The SQLite journal supports resuming by
case.

The task-001 oracle checks only whether `out/linecount.txt` is exactly `4` and
returns either 0 or 1. It does not measure input preservation. Therefore its
partial synthetic control deliberately retains the correct count while
modifying the input: the upstream oracle correctly assigns it the same perfect
score as the good output. The LLM rubric must lower the partial-control score
for the damaged input and distinguish it from the missing output. This is
reported as a coverage limitation of the deterministic oracle, not hidden by
changing its result.

## Analysis and reporting

The primary result is the LLM rubric outcome. The original continuous oracle
score, existing process score, and security score appear separately. The
`LLM outcome x process x security` product is a diagnostic. Missing judgments
or source values remain missing.

Language comparisons use task-balanced paired Hindi-English and
Hinglish-English differences. The same task indices are sampled together in
20,000 task-cluster bootstrap draws with seed 1701. The report gives paired
task counts, criterion scores, disagreement from the old oracle, judge usage,
review-needed cells, the known missing cell, traces, and workspace links. It
includes task and category summaries, score plots, and a 12-cell bilingual
review packet. Review selection is deterministic, with four cells per
language, four per harness, at least two per task, and coverage of low, middle,
and high LLM-oracle disagreement. The unblinding key is stored separately
from both reviewer forms and the blinded packet.

The PDF and machine-readable outputs remain local and ignored. Findings are
retrospective and exploratory: only five purposively selected tasks and one
execution per condition are represented; the same model family created and
judged agent work; translation quality and judge fairness await independent
bilingual review. This pilot is not directly numerically comparable with the
original Harness-Bench leaderboard.

## Commands

```powershell
py -3.14 -m runner.cli outcome-preflight --config configs/phase1.corrected.pilot-v14.yaml
py -3.14 -m runner.cli calibrate-outcomes --config configs/phase1.corrected.pilot-v14.yaml
py -3.14 -m runner.cli judge-outcomes --config configs/phase1.corrected.pilot-v14.yaml --resume
py -3.14 -m runner.cli outcome-report --config configs/phase1.corrected.pilot-v14.yaml
```

The last command refuses to report until all 45 source cells and traces are
indexed, 44 workspaces have valid two- or three-pass judgments, the one known
missing cell is identified, calibration matches all frozen identities, and
the secret and provenance checks pass.
