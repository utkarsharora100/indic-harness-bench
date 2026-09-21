# Architecture

```text
Task definition
     |
     v
Language variant
     |
     v
Experiment runner
     |
     +----------------------+
     |                      |
     v                      v
   Agent adapter           Fresh workspace
     |                      |
     +----------+-----------+
                |
                v
        Oracle + process judge
                |
                v
         SQLite + JSONL
                |
                v
             Analysis
```

## Package boundaries

`benchmark` owns task definitions and validation.

`agents` owns the common adapter interface, benchmark tools, the ReAct baseline,
and pinned native harness containers.

`runner` owns orchestration, per-run workspace creation, grading, and persistence.

`analysis` reads persisted runs and computes measurements.

`scripts` contains import and data-preparation commands.

## Workspace isolation

Every run starts from a clean copy of the task workspace. In Docker mode the copy is mounted into a fresh container. File operations are performed against the mounted workspace; shell commands and the deterministic grader execute inside the container.

The corrected study does not use host-process fallbacks for NanoBot or
OpenClaw. Native harness images receive only the workspace and a narrow
OpenAI-compatible proxy route. The university endpoint and bearer key remain
outside agent containers.

## Trace

The ReAct and native adapters record externally observable actions, tool
arguments/results where available, step numbers, and aggregate usage. The
post-run process judge receives a normalized observable trace and never private
chain-of-thought.
