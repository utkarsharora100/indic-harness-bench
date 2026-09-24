# Phase I outcome-v8: frozen reference-guided rejudgment

This version rejudges only the 44 archived, completed v14 pilot workspaces. The NanoBot/Hinglish/task-050 agent infrastructure failure remains missing. No agent is rerun, and neither the v14 database/PDF nor outcome-v7 is modified. The source database and original PDF are verified by their pinned SHA-256 hashes before and after each stage.

## Comparison contract

Each judge call has three distinct sections: the canonical English task question, numbered dataset reference requirements, and observations from the agent's saved final artifacts. The judge does not see the old oracle score, harness, prompt language, or raw input tables. Task 016 is the exception to the raw-input exclusion: submitted code and independently executed pytest results are shown, because a unique reference patch does not exist. Reference text never counts as submitted evidence.

The five tasks retain the v7 parent criterion weights. Every frozen requirement inside a criterion has equal weight. Exact checks cover values, IDs, status, hashes, and structured fields and permit level 0 or 4. Semantic checks allow levels 0–4, from absent/contradicted to fully correct. The model supplies a level, submitted evidence ID, and concise observation for every requirement; code validates the response and computes the final arithmetic. It does not invent or replace a level. Missing required artifacts cannot earn credit. Contradictions or unsupported evidence cause a retry or explicit review status, never an imputed zero.

One rating is followed by a focused verification of high-credit claims using the same model. Challenged claims receive a final resolution call. These are correlated checks, **not** independent judges. Technical call attempts, proxy events, usage, packet hashes, and frozen model/protocol identities are stored locally in append-only tables. Changing the scoring, packet, parser, model, or prompts requires a new outcome version.

## Sequence

1. Run `python -m runner.cli calibrate-outcomes-v8`. It verifies the source and 44 archives, uses task-specific synthetic workspaces (29 original controls plus 10 near-miss/alternative controls), and starts with the `001-file` live structural canary. A transport/schema error stops the batch. Calibration is resumable by case.
2. Run `python -m runner.cli judge-outcomes-v8`. It scores the same 44 archives in a seed-1701 shuffle and indexes the known missing cell. It never invokes an agent.
3. Run `python -m runner.cli outcome-report-v8`. The report independently recomputes scores from stored requirement levels, requires all 45 source cells to be classified, and writes ignored CSV/JSON tables and the provisional PDF under `output/pdf/`.

Substantive failed calibration controls are displayed prominently and downgrade the report to descriptive, judge-unvalidated results. Technical judge failures are reported as missing. Language claims remain exploratory until blinded bilingual human review. Original oracle scores are an audit comparison only. The pilot contains five purposively selected tasks and one execution per harness-language condition; it is not a direct replication of the original Harness-Bench leaderboard.
