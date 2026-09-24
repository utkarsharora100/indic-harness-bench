# Phase I main study, LLM-only outcome version 2

This protocol freezes a new experiment; it does not continue or pool with the
corrected pilot or the earlier 216-run experiment. Its matrix is 24 pinned tasks
× 3 prompt languages × 3 harnesses × 1 fresh execution = 216 cells. Task order
and the within-task condition order are reproducibly shuffled from seed 1701;
execution is sequential to avoid university-GPU contention. A task is the unit
of resampling, not an individual run.

## Conditions and provenance

The task source is Harness-Bench commit
`1025086a446653702b80cfb48babbeec35db6b2c`; task selection and all 72 English,
Hindi, and Hinglish prompts are pinned by the v13 selection and translation
manifests. The model, runtime image IDs, proxy, task tools, generation settings,
and code hashes are recorded in the ignored main24-v1 run manifest. Runs use the
v14 generation settings (temperature 0, top-p 1, 8192-token request ceiling,
40 proxy calls and at most 40 agent steps) with each task's upstream timeout.
The private endpoint credentials, URL, and served model identifier remain local.

## Outcomes

The primary task outcome is the university LLM judge's semantic score alone,
normalized as its 0–4 rating divided by four. The deterministic upstream task
oracle is retained as an audit comparison only and does not contribute to the
primary score. The semantic judge receives the canonical English requirement, reference facts, final
submitted artifacts, and independent tests for code tasks. It does not receive
the language, harness, oracle score, or unrelated raw input tables. Missing
deliverables are not credited based on the reference alone. Task 001's expected
line count is computed from its pinned fixture; task 087 uses task requirements
and independent tests because it has no single answer key. The semantic model
rating is a component, not an independent adjudicator; language findings remain
provisional pending bilingual review.

Process/security ratings are stored separately and are not folded into the
primary outcome. Any outcome × process × security product is descriptive only.
An oracle exception, malformed score, changed model/image, or incomplete trace is
an infrastructure failure, not a task failure.

## Gates and execution order

The study must not start before all 24 source/hook/fixture/oracle/workspace checks,
all judge controls, the all-condition nine-cell task-050 smoke, and the archived
44-run pilot rejudge pass under the main24 rubric. Corrections that alter scoring
or prompts require a new version and repeated affected gates. The 050 smoke is
stored separately and never counted among the 216 main cells. The long run saves
each attempt immediately and resumes only cells with matching frozen identity.
After all 216 cells are gradable, outcome judging runs automatically, followed
sequentially by process/security judging. No final findings report is produced
until explicitly requested; at that point report generation is the remaining
research step.

## Interpretation

The task subset is purposively selected, each condition has one execution, and
translations have not been independently reviewed. The same university model
provides semantic judgments. These are exploratory constraints, not evidence of
equivalence or causal superiority. Results will show task-balanced means,
paired language differences with task-cluster bootstrap intervals, harness
contrasts, all missingness, and trace-level provenance. Raw results stay in
ignored local storage; only protocol and implementation are tracked.
