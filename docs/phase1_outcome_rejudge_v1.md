# Phase I pilot outcome rejudgment - v1

## Purpose and version boundary

This protocol re-scores only the saved v14 pilot workspaces. It does not rerun
agents and does not alter the v14 run database, traces, archives, oracle grades,
or report. A separate ignored SQLite store is keyed by each original `run_id`.
The target is 44 archived runs from 45 planned cells; the NanoBot/Hinglish
`050-multitable-join-analysis` infrastructure error remains unscored.

This is a retrospective change to the primary outcome and is exploratory. The
five tasks were purposively selected, and there is one execution per condition.
Translations have not received independent bilingual review. The same pinned
university model family was used as agent and is used as outcome judge, so
self-preference and language bias are not ruled out. This is not directly
comparable to the original Harness-Bench leaderboard: the upstream paper uses
executable checks when possible and rubric scoring when needed, whereas this
rejudgment deliberately uses a single LLM rubric as the primary score.

## Outcome rubric

Each task has the frozen criteria and weights in
`configs/outcome-rubric.pilot-v14-v1.yaml`. The judge assigns each criterion
one ordinal level: 0 absent/contradicted, 1 attempted with major errors, 2
useful partial work, 3 substantially correct with minor omissions, or 4 fully
satisfying the requirement. Software computes the weighted score as
`sum(weight * level / 4)`; the model does not calculate the final score.
Criterion levels, rationales, and citations are retained. The old deterministic
oracle score remains an independent comparison column and is not shown to the
outcome judge.

## Evidence and blinding

For each cell, the judge receives the canonical English requirements, the
task's reference facts, task `in/` and `out/` files, and (for task 016) an
independent rerun of the public pytest command in an offline Docker container.
The archive path is confined to the experiment's workspace-archive directory;
the archive tree hash must match the original run record. Unsafe paths, links,
oversized files, and corrupt archives are rejected. Harness bootstrap files
outside `in/` and `out/` are excluded to reduce identity leakage. Evidence is
never silently truncated; oversized packets become explicit missing evidence.

The outcome packet omits harness, language, agent, old oracle score/pass list,
and process rating. It does not include private chain-of-thought. File content
is untrusted data, not evaluator instruction. Alternative wording is accepted
when requirements allow it. Because output text itself may reveal language,
condition blinding is not perfect.

## Repeated judgments and missingness

Each eligible run receives two independent calls at temperature 0 and top-p 1;
the second call reverses criterion order. Format and transient endpoint errors
are retried at most twice after the initial attempt and are retained in the
separate store with response, validation error, usage, and request/evidence
hashes. A third judgment is requested if overall scores differ by more than
0.15 or any criterion differs by at least two levels. Criterion medians are
used after a third call. If the resulting three-score spread exceeds 0.20,
the run is flagged `needs_review` and receives no primary score. Failed,
missing, corrupt, and review-needed judgments remain missing, never zero.

The resolved model identity is matched against the existing ignored v14 model
manifest before calibration or scoring. Its raw identity, endpoint, and bearer
key are not written to tracked files or report output. The outcome store
manifest pins source database, rubric, calibration, model hash, generation
settings, and shuffle seed; a mismatch requires a new versioned store.
The response-token ceiling follows the v14 experiment configuration because
the university endpoint rejects per-call caps that differ from that frozen
setting; the judge typically ends its JSON response well below that ceiling.

## Calibration gate

Before judging saved runs, execute `calibrate-outcomes`. It evaluates 29
synthetic cases: correct, partial, incorrect, missing-artifact, and prompt
injection controls for each task, plus Hindi and Hinglish equivalents for two
open-ended tasks. The gate checks ordinal separation, injection invariance,
exact line-count behavior, and language-equivalent scores within 0.10. Task
016 controls also record actual offline pytest outcomes. Any failed control
blocks rejudgment; a changed rubric or evidence packer requires a new version
and a complete rerun under that version.

```powershell
py -3.14 -m runner.cli calibrate-outcomes --config configs/phase1.corrected.pilot-v14.yaml
py -3.14 -m runner.cli judge-outcomes --config configs/phase1.corrected.pilot-v14.yaml --resume
py -3.14 -m runner.cli outcome-report --config configs/phase1.corrected.pilot-v14.yaml
```

Calibration records and the rejudgment database are local and ignored under
`data/phase1/corrected/pilot-v14/rejudgments/outcome-v1/`.

## Analysis and human review

The primary summary is the task-balanced mean of the new LLM outcome score.
Paired Hindi-English and Hinglish-English task differences use the same tasks
within each harness and exploratory 20,000-draw task-cluster bootstrap
intervals. The report also gives per-cell and per-task scores, oracle
disagreements, old process/security scores in separate columns, and links to
every trace and archived workspace. Any legacy outcome x process x security
composite is diagnostic only; it is not substituted for the primary outcome.

A deterministic stratified sample of 12 blinded evidence packets and a blank
review form are exported for two bilingual reviewers. They should rate
independently with the same rubric, cite evidence, and compare only after both
forms are complete. Until this review is returned and analyzed, no definitive
language or judge-fairness claim is warranted.

## Reference

- Harness-Bench paper: <https://arxiv.org/html/2605.27922>.
- LLM-as-a-judge position, verbosity, and self-preference bias study:
  <https://arxiv.org/abs/2306.05685>.
