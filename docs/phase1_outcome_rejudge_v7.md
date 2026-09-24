# Phase I v14 pilot: outcome-v7 rejudgment protocol

V7 supersedes incomplete v5 and v6 calibration attempts. Neither attempt judged
any saved agent workspace. V5 stopped after the task-016 missing-work control
received unusable output. V6 exposed the served model's stable response form:
it returned the correct criterion IDs with bare integer levels, while omitting
the requested object wrapper and reasons. Both attempt stores remain separate
for audit. V7 accepts compact integer ratings and optional reasons, and marks
missing reasons explicitly in the evidence tables.

The model receives canonical English requirements, dataset reference answers,
and submitted workspace files in distinct fields. The answer key can establish
correctness but cannot prove delivery. Code attaches artifact paths, hashes,
reference-field names, and independent task-016 test results. Code assigns
scores for absent required artifacts and fixture-preservation checks, and
omits those criteria from the model request. For task-016, unchanged broken
code cannot receive repair credit and failing tests cap a modified repair at
partial credit. The model judges the remaining submitted work on the frozen
0-4 scale; the program applies fixed criterion weights.

Calibration is resumable and append-only. Offline preflight checks the frozen
24 task sources, 44 archives, 45 traces, nine task-016 test runs, and 29
synthetic controls. A live transport smoke and five varied structural cases
run before the remaining controls. Invalid numeric levels, missing criterion
IDs, malformed JSON, or endpoint failures stop calibration; valid bare integer
ratings are accepted. Fully contract-resolved synthetic cases are recorded as
deterministic scores without model calls. Substantive calibration failures
remain visible in any report and make its conclusions descriptive and
judge-unvalidated.

The only agent evidence is the 44 completed v14 workspaces; one known missing
agent-artifact cell remains missing. The original source database and PDF
hashes are verified before and after every stage. V7 judgments use two passes,
with a third for the frozen disagreement thresholds. Technical errors remain
missing, never zero. Any findings are retrospective, self-judged, based on
five purposively selected tasks and one execution per condition; translations
still need bilingual review. The main study remains out of scope.
