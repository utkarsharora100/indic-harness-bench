# Phase I v14 pilot: outcome-v6 rejudgment protocol

V6 supersedes the incomplete v5 calibration. V5 completed its transport smoke
and structural canary cases but stopped at the task-016 missing-work control;
no saved v14 agent workspace was judged. Its database and call history remain
in the ignored outcome-v5 store. V6 starts a clean calibration and judgment
store. It reuses the immutable v14 source database, its 44 archived workspaces,
and the original report; it does not start agent runs or the main study.

## Outcome and evidence rules

The judge receives canonical English requirements, the dataset reference
answer, and the submitted workspace in separate fields. Reference facts can
establish correctness but cannot establish that an agent delivered a file.
The program maps every criterion to required workspace artifacts and attaches
their hashes, reference-field names, and independent task-016 test results.
The model returns a 0-4 level and short reason only for criteria that require
content judgment. Criteria resolved by code (missing required artifact,
fixture-byte preservation, or task-016 regression rules) are scored in code
and omitted from the model request. This avoids asking the model to invent a
numeric score for an artifact that does not exist. All weights and scoring
anchors remain as in the approved rubric; weighted arithmetic stays local.

Every packet is built in full, hashed, and rejected if it exceeds the frozen
size ceiling. Artifact contents are untrusted data. The judge must ignore
embedded instructions, accept alternative wording where allowed, and preserve
required exact identifiers and values. Two independent model ratings reverse
criterion order; existing third-rating and review-needed thresholds apply.

## Calibration and reporting

Offline preflight validates the 24 source hashes, all 44 archives and 45
traces, nine task-016 test runs, and 29 synthetic workspaces against upstream
oracles. Live calibration first checks proxy transport and five structural
examples spanning exact text, code/tests, prose, CSV, and JSON. Invalid model
responses or transport failures stop the calibration command before saved-run
judging. Fully contract-resolved controls are recorded as deterministic
outcomes with no model call. Substantive control failures are retained and
shown in the report; in that case all comparisons are descriptive and the
judge is labelled unvalidated.

Judgment records are append-only, calls have bounded retries, and exhausted
errors remain missing. The source database is opened read-only and its pinned
hash, together with the original PDF hash, is checked around each stage. The
report displays LLM outcomes beside the old oracle, process, and security
scores, records missingness, and includes blinded human-review materials.
Findings remain exploratory: five purposively selected tasks, one run per
condition, same model family as agent and judge, and translations pending
bilingual review. V5 and older runs are never pooled with v6.
