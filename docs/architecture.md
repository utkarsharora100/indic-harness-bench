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
              Grader
                |
                v
         SQLite + JSONL
                |
                v
             Analysis
```

## Package boundaries

`benchmark` owns task definitions and validation.

`agents` owns the common adapter interface, benchmark tools, the ReAct baseline, and external runtime wrappers.

`runner` owns orchestration, per-run workspace creation, grading, and persistence.

`analysis` reads persisted runs and computes measurements.

`scripts` contains import and data-preparation commands.

## Workspace isolation

Every run starts from a clean copy of the task workspace. In Docker mode the copy is mounted into a fresh container. File operations are performed against the mounted workspace; shell commands and the deterministic grader execute inside the container.

External host runtimes such as NanoBot and OpenClaw are wrapped as subprocess adapters. They are not presented as a Docker security boundary by this repository; use the corresponding native Harness-Bench adapters for a fully controlled production evaluation.

## Trace

The ReAct adapter records externally observable tool calls, arguments, results, step numbers, and aggregate usage. Private chain-of-thought is not stored.
