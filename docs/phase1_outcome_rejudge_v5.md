# Phase I v14 pilot: outcome-v5 rejudgment protocol

This version reuses the 44 completed v14 agent workspaces; the NanoBot/Hinglish
task-050 infrastructure cell remains missing. It does not launch agents or the
648-cell main study. Earlier outcome-v1/v3/v4 attempts stay in their separate
ignored stores and are never pooled. The source database and original PDF must
retain their pinned SHA-256 hashes (checked before and after each command).

## Why v5 exists

V4 calibration rated the task-019 missing-output control 1.0 by citing the
reference answer and input runbooks, despite no submitted outputs and an oracle
score of zero. It also repeatedly emitted malformed citation locators. V5
separates canonical requirements, reference answers, and submitted artifacts.
The answer key may establish correctness but never proves delivery. The model
returns only criterion levels and concise reasons; code attaches the frozen
artifact paths, reference fields, hashes, and independent task-016 tests.

The approved task criteria and weights remain unchanged. Code fixes absent
required outputs at level 0, checks supplied-input preservation by fixture
hash, and caps an untested/broken task-016 repair. Model judgment compares the
remaining submitted content against task-specific reference facts. Equivalent
wording is valid where the task permits it; required identifiers and numeric
values are exact. No evidence is silently truncated. The outcome is the
weighted mean of 0-4 criterion levels divided by four.

## Gates and interpretation

`outcome-preflight` is offline: it checks 24 source hashes, 44 archives, 45
traces, nine archived task-016 tests, 29 synthetic workspaces, and upstream
oracle ordering. `calibrate-outcomes` is the first live-model command. It runs
a transport smoke and five structural canaries before the remaining controls.
Format/transport failure stops calibration without scoring saved runs.
Substantive failed controls are preserved as `failed_substantive_controls` and
may be followed by judging, but the eventual report must prominently label
all comparisons descriptive and judge-unvalidated. Technical judgment errors
remain missing; they are not zeros.

The source SQLite database is opened read-only. Two model ratings per saved
workspace reverse criterion order; a third is requested for a score gap over
0.15 or a criterion gap of at least two levels. The final criterion level is
the median of three, or mean of two. A three-pass score span over 0.20 leaves
the cell `needs_review` with no primary score. All attempts, usage, redacted
responses, proxy events, packet hashes, and verdicts are retained. The same
pinned university model is both agent and outcome judge, so the judge is not
independent. Hindi/Hinglish translations and a bilingual 12-run review packet
remain unapproved; no definitive language claim is permitted.

Protocol identity hashes judge code, rubric, evidence mapping, model and
settings. Report/PDF source hashes are separate, so presentation changes do
not invalidate completed ratings. Local results and PDF are git-ignored. Do
not publish until review and publication decisions are made.
