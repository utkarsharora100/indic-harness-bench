# Indic-Harness-Bench — Phase 1 Plan and Architecture

## 1. Phase 1 Goal

The goal of Phase 1 is to measure whether the **language of a user instruction** affects an AI agent's ability to complete the **same executable task**.

We compare three instruction languages:

- **English**
- **Hindi (Devanagari)**
- **Hinglish (Romanized Hindi + English)**

The task, model, agent framework, tools, sandbox, execution budget, and grading procedure remain fixed wherever possible. The main experimental variable is the instruction language.

### Core Research Question

> Does changing only the instruction language from English to Hindi or Hinglish reduce an agent's ability to successfully complete a realistic software/workflow task?

### Secondary Questions

1. Is performance degradation larger for Hindi or Hinglish?
2. At which stage do failures occur?
   - instruction understanding
   - planning
   - tool selection
   - command/tool argument generation
   - execution
   - context/state tracking
   - error recovery
   - final completion
3. Does language affect efficiency even when the task is successfully completed?
4. Do different agent harnesses behave differently under Hindi and Hinglish instructions?

---

# 2. Core Experimental Principle

Phase 1 follows a **controlled paired-task design**.

For one task:

```text
Same Task
   |
   +---- English instruction
   |
   +---- Hindi instruction
   |
   +---- Hinglish instruction
```

Everything else stays the same.

### Controlled Variables

- task
- task files
- workspace
- input data
- tools
- model
- model version/checkpoint
- system prompt
- agent configuration
- temperature
- top-p
- maximum tokens
- maximum agent steps
- timeout
- grading tests
- sandbox base image

### Independent Variable

```text
Instruction language
```

### Dependent Variables

```text
Success rate
Execution time
Token usage
Number of tool calls
Number of failed tool calls
Number of agent steps
Recovery behavior
Failure category
```

---

# 3. Phase 1 Architecture

```text
                    ┌─────────────────────────────┐
                    │     Canonical Task Set      │
                    │  Harness-Bench task subset  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    Task Variant Generator    │
                    │                             │
                    │ English / Hindi / Hinglish  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │     Experiment Controller    │
                    │                             │
                    │ task × language × agent ×   │
                    │ model × repetition          │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       Agent Adapter          │
                    │                             │
                    │ ReAct / NanoBot / OpenClaw  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      Isolated Sandbox        │
                    │                             │
                    │ files + tools + environment │
                    └──────────────┬──────────────┘
                                   │
                       ┌───────────┴───────────┐
                       ▼                       ▼
              ┌─────────────────┐      ┌─────────────────┐
              │ Execution Logger│      │ Deterministic   │
              │                 │      │ Judge           │
              │ actions/traces  │      │ pytest/checks   │
              └────────┬────────┘      └────────┬────────┘
                       │                        │
                       └───────────┬────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │       Results Database       │
                    │ SQLite + JSONL traces       │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      Analysis Pipeline       │
                    │ metrics + paired failures   │
                    │ language comparison         │
                    └─────────────────────────────┘
```

---

# 4. Main Components

## 4.1 Canonical Task Set

Start with a small, representative subset of Harness-Bench tasks rather than the full benchmark.

### Recommended Initial Size

**24 tasks**

Suggested distribution:

| Category | Tasks |
|---|---:|
| Software engineering / code maintenance | 6 |
| Data / file processing | 5 |
| Shell / tool use | 4 |
| Office / business workflow | 4 |
| Knowledge / retrieval | 3 |
| Multi-step workflow | 2 |
| **Total** | **24** |

The purpose is to test agent behavior beyond simple code generation.

---

# 5. Canonical Task Representation

Each task should have exactly one canonical definition.

Example:

```yaml
task_id: HB_SE_004
category: software_engineering

instruction:
  english: "Fix the failing database query and ensure all tests pass."
  hindi: "डेटाबेस की असफल query को ठीक करें और सुनिश्चित करें कि सभी tests पास हों।"
  hinglish: "Database ki failing query ko fix karo aur ensure karo ki saare tests pass ho."

environment:
  image: indic-harness/python-db:1.0

workspace:
  source: tasks/HB_SE_004/workspace

limits:
  max_steps: 30
  timeout_seconds: 600

judge:
  type: pytest
  command: pytest grader/test_task.py
```

### Important Rule

The following should remain identical across language variants:

- files
- file contents
- filenames
- directory structure
- commands available
- test suite
- environment
- task goal

Only the natural-language instruction should change.

---

# 6. Language Variant Creation

Do not directly rely on raw machine translation.

Use the following pipeline:

```text
Original English Task
        │
        ▼
Initial Hindi/Hinglish Translation
        │
        ▼
Human Review
        │
        ▼
Back-translation / Semantic Check
        │
        ▼
Technical-Term Consistency Check
        │
        ▼
Final Approved Variant
```

## 6.1 Hindi

Use **Devanagari Hindi** for the primary Hindi condition.

Example:

```text
कोड में मौजूद त्रुटि को ठीक करें और सभी परीक्षण सफल होने सुनिश्चित करें।
```

Technical identifiers should remain unchanged when necessary:

```text
pytest
README.md
API
JSON
CSV
SQL
Git
```

## 6.2 Hinglish

Use natural technical Hinglish rather than forced word-by-word translation.

Example:

```text
Code mein jo error aa raha hai usko fix karo aur ensure karo ki saare tests pass ho rahe hain.
```

The benchmark should define a small style guide so different tasks use reasonably consistent Hinglish conventions.

---

# 7. Protect Technical Entities During Translation

Technical entities should not accidentally change during translation.

Protect items such as:

```text
file names
directory paths
commands
function names
variable names
URLs
package names
numbers
identifiers
API names
SQL keywords
```

Example:

```text
Original:
Run `pytest grader/test_task.py` and fix the error in `src/db.py`.

Protected:
Run <CMD_1> and fix the error in <FILE_1>.
```

Translate the surrounding sentence and restore the protected entities afterward.

### Validation

Automatically verify:

```text
same filenames
same commands
same identifiers
same numbers
same paths
```

A translation should not enter the benchmark until this validation passes.

---

# 8. Agent Adapter Architecture

All agent frameworks should expose the same benchmark interface.

```python
class AgentAdapter:
    def initialize(self, model, tools, config):
        ...

    def run(self, instruction, workspace):
        ...

    def get_trace(self):
        ...

    def get_usage(self):
        ...

    def shutdown(self):
        ...
```

Implement adapters such as:

```text
agents/
├── base.py
├── react.py
├── nanobot.py
└── openclaw.py
```

### Why Use an Adapter?

The experiment runner should not need to know the internal implementation of each framework.

It should simply call:

```python
agent.initialize(...)
agent.run(...)
agent.get_trace()
agent.get_usage()
agent.shutdown()
```

This makes framework comparisons reproducible.

---

# 9. Baseline Agent

Start with **ReAct** as the first baseline.

Conceptually:

```text
Instruction
     ↓
Reason about next action
     ↓
Choose tool
     ↓
Execute tool
     ↓
Observe result
     ↓
Decide next action
     ↓
Repeat
     ↓
Finish task
```

The benchmark does not need to expose private chain-of-thought.

Instead, record externally observable events such as:

```text
tool selected
tool arguments
tool result
file changes
command execution
errors
recovery actions
final state
```

---

# 10. Experiment Separation

Do not mix the language experiment and framework experiment.

## Experiment A — Language Effect

Fix:

```text
Agent = ReAct
Model = fixed
Task = fixed
Environment = fixed
```

Change only:

```text
English
Hindi
Hinglish
```

This is the **main Phase 1 experiment**.

### Purpose

Measure:

> What happens when the same agent receives the same task in different languages?

---

## Experiment B — Harness Interaction

After Experiment A is stable, compare:

```text
ReAct
NanoBot
OpenClaw
```

under each language.

This answers:

> Does the effect of Hindi/Hinglish depend on the agent framework?

---

# 11. Sandbox Design

Every execution must use a fresh isolated environment.

```text
Base Docker Image
        │
        ├── Run 1 → Fresh Sandbox
        ├── Run 2 → Fresh Sandbox
        ├── Run 3 → Fresh Sandbox
        └── ...
```

Never reuse the modified workspace from another language condition.

### Example

For task `HB_SE_004`:

```text
HB_SE_004/
├── base_workspace/
├── grader/
└── runs/
    ├── english_run_01/
    ├── hindi_run_01/
    └── hinglish_run_01/
```

This prevents contamination.

---

# 12. Experimental Fairness Rules

The agent should see the same environment regardless of language.

Keep fixed:

```text
System prompt
Tool descriptions
Available tools
Workspace
Model checkpoint
Model parameters
Execution budget
Timeout
Grader
Sandbox image
```

The main experiment changes only:

```text
USER INSTRUCTION
```

The system prompt can remain English because the research motivation is specifically to test Hindi/Hinglish instructions inside a predominantly English software environment.

---

# 13. Tool Set

Use a small and controlled tool set in the initial implementation.

Example:

```text
read_file
write_file
list_files
search_files
run_command
python
```

The available tools must be identical for English, Hindi and Hinglish runs.

---

# 14. Experiment Controller

The experiment controller generates all required runs from a configuration file.

Example:

```yaml
experiment:
  name: phase1_language_comparison

tasks:
  - HB_SE_001
  - HB_SE_002
  - HB_SE_003

languages:
  - english
  - hindi
  - hinglish

agents:
  - react

models:
  - model_x

repetitions: 3

generation:
  temperature: 0
  top_p: 1.0

limits:
  max_steps: 30
  timeout_seconds: 600
```

Run command:

```bash
python -m runner.experiment configs/phase1.yaml
```

---

# 15. Experimental Runs

## Pilot

Before the full experiment:

```text
5 tasks × 3 languages × 1 agent × 1 model × 1 run
= 15 executions
```

Use the pilot to find:

- broken tasks
- ambiguous translations
- grading problems
- tool integration issues
- unstable agent behavior
- logging problems

Only after the pilot passes should the benchmark scale.

---

## Main Phase 1

Recommended initial experiment:

```text
24 tasks
× 3 languages
× 1 agent
× 1 model
× 3 repetitions
```

Total:

```text
24 × 3 × 1 × 1 × 3 = 216 runs
```

---

## Extended Phase 1

After the language-only experiment is stable:

```text
24 tasks
× 3 languages
× 3 agents
× 1 model
× 3 repetitions
= 648 runs
```

This adds the framework dimension.

---

# 16. Run Randomization

Do not always execute in this order:

```text
English → Hindi → Hinglish
```

Randomize run order.

Example:

```text
Task 07 Hinglish
Task 12 English
Task 03 Hindi
Task 07 English
Task 12 Hinglish
...
```

This reduces the risk of results being affected by:

- machine load
- GPU contention
- cache effects
- temporary resource variation

---

# 17. Logging Architecture

Every run should produce two levels of data.

## 17.1 Summary Data

Store in SQLite.

```text
run_id
task_id
language
agent
model
seed
start_time
end_time
success
input_tokens
output_tokens
total_tokens
tool_calls
failed_tool_calls
execution_time
```

## 17.2 Detailed Trace

Store in JSONL.

Example event:

```json
{
  "run_id": "run_00017",
  "step": 4,
  "event_type": "tool_call",
  "tool": "run_command",
  "arguments": "pytest grader/test_task.py",
  "result": "2 failed, 8 passed",
  "timestamp": "2026-09-09T10:15:22"
}
```

---

# 18. Database Schema

Recommended tables:

## TASK

```text
task_id
category
source_task
```

## LANGUAGE_VARIANT

```text
task_id
language
instruction
review_status
reviewer
```

## RUN

```text
run_id
task_id
language
model
agent
seed
start_time
end_time
success
input_tokens
output_tokens
total_tokens
tool_calls
failed_tool_calls
execution_time
```

## EVENT

```text
event_id
run_id
step
event_type
tool
arguments
result
timestamp
```

## GRADE

```text
run_id
test_name
passed
details
```

## FAILURE

```text
run_id
failure_category
failure_subcategory
annotator
notes
```

---

# 19. Deterministic Grading

Success should be determined from the **final environment state**, not from the agent's final message.

Example:

```bash
pytest grader/test_task.py
```

### Success Definition

A run is successful when all mandatory grader checks pass.

```text
Success = 1  if all required tests pass
          0  otherwise
```

This avoids subjective evaluation of final text.

---

# 20. Primary Metric

## Success Rate

For language \(L\):

\[
SR_L = \frac{\text{Number of successful runs in language }L}
{\text{Total runs in language }L}
\]

Calculate:

```text
SR_English
SR_Hindi
SR_Hinglish
```

---

# 21. Language Degradation

Hindi relative to English:

\[
\Delta_{Hindi} = SR_{Hindi} - SR_{English}
\]

Hinglish relative to English:

\[
\Delta_{Hinglish} = SR_{Hinglish} - SR_{English}
\]

A negative value indicates performance degradation.

You can also report:

\[
Degradation_{Hindi}
=
SR_{English} - SR_{Hindi}
\]

\[
Degradation_{Hinglish}
=
SR_{English} - SR_{Hinglish}
\]

---

# 22. Secondary Metrics

Measure:

### Efficiency

```text
execution time
total tokens
agent steps
tool calls
```

### Tool Reliability

\[
ToolErrorRate =
\frac{\text{Failed tool calls}}
{\text{Total tool calls}}
\]

### Recovery

Measure:

```text
Did the agent recover after an error?
How many additional actions were required?
Did the task eventually succeed?
```

A useful metric is:

\[
RecoveryRate =
\frac{\text{Runs that recover after an error}}
{\text{Runs that encounter an error}}
\]

---

# 23. Failure Taxonomy

A major Phase 1 contribution is identifying **where the agent fails**.

## F1 — Instruction Understanding

Examples:

```text
F1.1 ignored constraint
F1.2 misunderstood Hindi phrase
F1.3 misunderstood Hinglish phrase
F1.4 wrong entity interpretation
F1.5 incorrect interpretation of technical wording
```

## F2 — Planning

Examples:

```text
wrong plan
missing required step
incorrect task decomposition
```

## F3 — Tool Selection

Examples:

```text
wrong tool
never called required tool
unnecessary tool calls
```

## F4 — Tool Argument / Command Generation

Examples:

```text
invalid command
wrong path
wrong argument
syntax error
```

## F5 — Execution

Examples:

```text
command failed
file operation failed
runtime error
```

## F6 — Context / State Tracking

Examples:

```text
forgot previous observation
used stale information
repeated completed action
lost track of changed file
```

## F7 — Error Recovery

Examples:

```text
ignored error
repeated failing command
wrong recovery action
gave up too early
```

## F8 — Final Completion

Examples:

```text
task partially completed
required artifact missing
final state incorrect
```

## F9 — Evaluation / Infrastructure

Examples:

```text
grader error
sandbox problem
logging failure
benchmark bug
```

---

# 24. Paired Failure Analysis

Because English, Hindi and Hinglish are different variants of the **same task**, failures can be compared directly.

Example:

```text
Task HB_SE_014

English    → PASS
Hindi      → FAIL
Hinglish   → PASS
```

Now inspect the traces.

Possible discovery:

```text
English:
instruction
  ↓
correct plan
  ↓
correct tool
  ↓
successful execution

Hindi:
instruction
  ↓
misinterprets "index"
  ↓
opens wrong file
  ↓
wrong command
  ↓
failure
```

This gives stronger evidence than simply reporting:

```text
Hindi success rate = 62%
```

The trace explains **why** the success rate changed.

---

# 25. First-Divergence Analysis

For failed runs, identify the earliest point at which behavior diverged from the expected successful trajectory.

Example:

```text
Expected:
read schema → inspect query → edit SQL → run tests

Observed:
read README → inspect unrelated file → wrong command
```

Label the first meaningful divergence:

```text
F1 Instruction Understanding
```

rather than labeling every downstream error as a separate root cause.

This makes failure analysis more meaningful.

---

# 26. Annotation Process

A subset of failure traces should be manually reviewed.

Recommended process:

```text
Two annotators independently label failures
                 ↓
Compare labels
                 ↓
Resolve disagreements
                 ↓
Create final taxonomy labels
```

Where feasible, report inter-annotator agreement for the manually labeled sample.

---

# 27. Reproducibility and Versioning

Every run should record:

```text
Git commit
Benchmark version
Task version
Docker image hash
Model name
Model hash/checkpoint
Agent version
Python version
Ollama version (if used)
Operating system
GPU/CUDA version
Experiment configuration
```

This allows an experiment to be reproduced later.

---

# 28. Recommended Repository Structure

```text
indic-harness-bench/
│
├── README.md
├── pyproject.toml
│
├── configs/
│   ├── phase1.yaml
│   ├── models.yaml
│   └── agents.yaml
│
├── benchmark/
│   ├── tasks/
│   │   ├── HB001/
│   │   │   ├── task.yaml
│   │   │   ├── workspace/
│   │   │   ├── grader/
│   │   │   └── translations/
│   │   │       ├── english.txt
│   │   │       ├── hindi.txt
│   │   │       └── hinglish.txt
│   │   └── HB002/
│   │
│   └── schemas/
│
├── agents/
│   ├── base.py
│   ├── react.py
│   ├── nanobot.py
│   └── openclaw.py
│
├── tools/
│   ├── filesystem.py
│   ├── shell.py
│   └── python_tool.py
│
├── runner/
│   ├── experiment.py
│   ├── sandbox.py
│   ├── logger.py
│   └── grader.py
│
├── analysis/
│   ├── metrics.py
│   ├── failure_analysis.py
│   ├── statistics.py
│   └── plots.py
│
├── data/
│   ├── runs.sqlite
│   ├── traces/
│   └── results/
│
├── docker/
│   ├── Dockerfile
│   └── compose.yaml
│
└── tests/
```

---

# 29. Phase 1 Implementation Sequence

## Step 1 — Select Tasks

Choose 5 Harness-Bench tasks for the pilot.

## Step 2 — Create Language Variants

Create:

```text
English
Hindi
Hinglish
```

for every selected task.

## Step 3 — Validate Variants

Check:

```text
semantic equivalence
technical identifiers
commands
filenames
paths
numbers
```

## Step 4 — Implement Canonical Task Format

Create `task.yaml` for every task.

## Step 5 — Build Sandbox

Make sure every run starts from a clean environment.

## Step 6 — Implement ReAct Adapter

Connect the first agent to the benchmark interface.

## Step 7 — Implement Logger

Record:

```text
model request
tool call
tool result
workspace changes
timing
token usage
errors
```

## Step 8 — Implement Deterministic Grader

Use task-specific tests.

## Step 9 — Run 15-Run Pilot

```text
5 tasks × 3 languages = 15 runs
```

## Step 10 — Debug Benchmark

Fix:

```text
translation issues
tool bugs
sandbox bugs
grader bugs
logging bugs
```

## Step 11 — Scale Language Experiment

```text
24 tasks × 3 languages × 3 repetitions
= 216 runs
```

## Step 12 — Analyze Results

Produce:

```text
success-rate comparison
token comparison
time comparison
tool-call comparison
failure taxonomy
paired trace analysis
```

## Step 13 — Add Framework Comparison

Add:

```text
NanoBot
OpenClaw
```

only after the core experiment is stable.

---

# 30. Expected Phase 1 Outputs

At the end of Phase 1, the project should produce:

### 1. Benchmark Dataset

```text
24 executable tasks
×
3 instruction languages
```

### 2. Reproducible Runner

A configuration-driven system that can run the same experiment repeatedly.

### 3. Agent Adapters

At minimum:

```text
ReAct
```

and later:

```text
NanoBot
OpenClaw
```

### 4. Execution Traces

Detailed records of agent actions and tool interactions.

### 5. Deterministic Evaluation

Task-level automated success/failure.

### 6. Language Comparison

```text
English vs Hindi
English vs Hinglish
Hindi vs Hinglish
```

### 7. Failure Analysis

Evidence showing **where and why** multilingual agents fail.

---

# 31. Final Phase 1 Architecture in One View

```text
                    INDIC-HARNESS-BENCH
                         PHASE 1
                            │
                            ▼
                ┌──────────────────────┐
                │   Canonical Tasks    │
                │ Harness-Bench subset │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Language Variants    │
                │                      │
                │ English              │
                │ Hindi                │
                │ Hinglish             │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Experiment Controller│
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    Agent Adapter     │
                │      ReAct           │
                │  NanoBot / OpenClaw  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Fresh Sandbox      │
                │                      │
                │ workspace + tools    │
                └──────────┬───────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
        ┌─────────────────┐  ┌─────────────────┐
        │ Trace Logger    │  │ Deterministic   │
        │                 │  │ Judge           │
        └────────┬────────┘  └────────┬────────┘
                 │                    │
                 └──────────┬─────────┘
                            ▼
                  ┌────────────────────┐
                  │ Results + Traces   │
                  │ SQLite + JSONL     │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Analysis           │
                  │                    │
                  │ Success            │
                  │ Tokens             │
                  │ Time               │
                  │ Tool calls         │
                  │ Recovery           │
                  │ Failure taxonomy   │
                  └────────────────────┘
```

---

# 32. One-Sentence Definition of Phase 1

> **Phase 1 is a controlled multilingual agent-evaluation experiment in which the same executable tasks are given in English, Hindi, and Hinglish, while the model, agent, tools, environment, budget, and grading remain fixed, enabling measurement of language-related effects on task success, efficiency, and failure behavior.**
