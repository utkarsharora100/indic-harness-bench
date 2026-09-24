# Phase I pilot outcome rejudgment v4

## Version boundary and audit history

Outcome-v2 stopped at offline deterministic calibration and made no live model
calls. Outcome-v3 passed deterministic checks and made one transport smoke call
plus two of its 29 LLM controls. Its task-019 missing-work control then
exhausted three attempts because the judge quoted a real runbook instruction
with changed capitalization. The v3 attempts remain in
`data/phase1/corrected/pilot-v14/rejudgments/outcome-v3/` as an audit trail;
they are not calibration passes and are not pooled with v4. No saved pilot
workspace was judged in v2 or v3.

V4 is the first version eligible to calibrate against saved pilot workspaces.
Any v4 calibration failure blocks that judging command. Outcome scores, v14
agent traces/workspaces, and the v14 report are never edited by this pipeline.

## Frozen inputs and rubric

The source v14 database SHA-256 is
`d62e4753eb45fec807427e6866d88a34966b78cfa2fc1815689d2e497cccfecb`; the
original PDF SHA-256 is
`bd5b26f3f93bba48167e8c339a9656e8232c61fd323d0d03ecc6b08e9ffd2780`.
Commands verify both before and after calibration, judging, report generation,
and PDF creation. The source database is queried with SQLite read-only mode.
Outcome commands never construct `ExperimentRunner` or `RunStore`.

`configs/outcome-rubric.pilot-v14-v4.yaml` retains the approved task criteria
and weights. Each criterion is rated on the frozen 0-4 scale; local code
calculates the weighted outcome score in [0,1]. The judge sees canonical
English requirements, task reference facts, final input/output/code files, and
independent task-016 pytest evidence. It does not see old oracle/process scores,
condition labels, or private chain-of-thought. Evidence is prebuilt, hashed,
size-checked, and never truncated. Each rating cites a stable evidence ID and
path, plus a structurally validated line, JSON pointer, test observation, or
verbatim quote.

## Citation retry correction

V3 correctly rejected a non-verbatim quote, but it retried the identical prompt
without explaining the validation failure. V4 explicitly instructs the model
to prefer line/JSON/test locators and to copy any quote exactly, including
case and punctuation. If response validation fails, the next bounded retry
receives the redacted validation error and is asked to repair only the JSON or
citation, not to change its substantive ratings. Transient timeouts, connection
failures, 429, and 5xx errors remain retryable; deterministic 4xx errors do not.
Three total attempts is the maximum. Every prompt hash, raw redacted response,
validation error, usage record, call event, and proxy event is preserved in
append-only tables. Repeated invalid responses block calibration or judging;
they never become a zero score.

## Calibration and score exceptions

Before any live judging, preflight verifies all 24 pinned task-source hashes,
the 45-cell v14 matrix, all 44 workspace archives and 45 traces, and nine
independent task-016 pytest runs. It grades 29 synthetic correct, partial,
incorrect, missing, prompt-injection, and equivalent-language workspaces in
offline Docker against upstream deterministic oracles. A single 001-file
correct-control call through the live proxy then verifies JSON, usage, public
alias routing, and redaction. All 29 LLM controls must pass ranking, exact
001-file checks, prompt-injection invariance, and Hindi/Hinglish output
equivalence checks before v4 calibration is frozen as passed.

Two inherited upstream-oracle limitations are explicitly controlled. The
001-file oracle checks only the exact line count `4`; its partial control also
preserves the correct count while mutating input, so this oracle ties that
partial control with correct work. The LLM criterion for input preservation
must distinguish them. The task-016 missing-work oracle score is 0.27 because
the untouched fixture earns 0.30 for an unchanged test-file hash then receives
a 0.90 multiplier for the broken implementation. Its frozen missing ceiling
is therefore 0.30; removing input fixtures to force a lower score would change
the upstream task contract.

## Rejudgment and analysis

Only after a matching passing v4 calibration may the command score the 44
archived saved runs, shuffled with seed 1701. Each receives two independent
ratings in reversed criterion order; a third is required if weighted scores
differ by more than 0.15 or any criterion differs by at least two levels.
Three ratings use per-criterion medians. If their weighted-score range exceeds
0.20, the result is `needs_review` and the primary score is missing. The
known NanoBot/Hinglish/task-050 cell remains `missing_agent_artifact`. No new
agent executions are started.

The report is blocked unless all 45 source cells and traces are indexed, the
44 available runs have valid two/three-pass histories, calibration and code
identities match, and exactly one known agent artifact is missing. It reports
LLM outcome as the primary retrospective measure; original oracle, process,
and security values stay separate. Their product is a diagnostic only.
Language comparisons are task-balanced paired differences with 20,000
task-cluster bootstrap draws at seed 1701. The local provisional PDF includes
all cell/criterion ratings, oracle disagreements, calibration, missing or
review-needed cells, charts, clickable trace/workspace references, and a
deterministically selected blinded 12-cell human-review packet.

The 12-cell human sample targets exactly four per language and harness, at
least two per task, and low/middle/high oracle disagreement coverage. The
unblinding key is a separate ignored file. Results remain exploratory: five
purposively selected tasks, one execution per condition, same model family for
generation and judging, and no independent bilingual validation yet. They are
not numerically comparable with the original Harness-Bench leaderboard.

## Run sequence

```powershell
py -3.14 -m pytest -q
py -3.14 -m runner.cli outcome-preflight --config configs/phase1.corrected.pilot-v14.yaml
py -3.14 -m runner.cli calibrate-outcomes --config configs/phase1.corrected.pilot-v14.yaml
py -3.14 -m runner.cli judge-outcomes --config configs/phase1.corrected.pilot-v14.yaml --resume
py -3.14 -m runner.cli outcome-report --config configs/phase1.corrected.pilot-v14.yaml
```

Artifacts are stored under ignored
`data/phase1/corrected/pilot-v14/rejudgments/outcome-v4/` and the v4 PDF is
`output/pdf/indic_harness_phase1_llm_judged_regrade_v4.pdf`. Only code, tests,
this protocol, and the frozen v4 rubric are intended for Git.
