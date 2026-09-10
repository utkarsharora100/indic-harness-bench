# Phase 1 experiment

## Language-only stage

The primary comparison fixes the task, model, agent, tool set, system prompt, environment, execution budget, and grader. The instruction changes between English, Hindi, and Hinglish.

With 24 tasks and 3 repetitions:

```text
24 × 3 languages × 3 repetitions = 216 runs
```

Run order is shuffled with a fixed seed.

## Harness stage

After the language-only stage is stable, compare ReAct, NanoBot, and OpenClaw. Report this as a separate analysis because changing harnesses changes more than one execution-layer property.

## Translation protocol

The English instruction is the canonical semantic source. Hindi and Hinglish variants should pass a human review and a protected-entity check.

Protect:

- filenames
- paths
- commands
- function and variable names
- URLs
- package names
- identifiers
- numbers

Do not translate technical identifiers unless the task explicitly asks the agent to modify them.

## Grading

Use deterministic task-local graders where possible. The preferred form is an executable command such as `pytest`. The grader should inspect the resulting workspace rather than the final prose response.

## Failure analysis

Classify the earliest meaningful divergence into instruction understanding, planning, tool selection, tool argument generation, execution, context/state tracking, error recovery, completion, or evaluation infrastructure.

For each task, compare the English, Hindi, and Hinglish traces directly.
