# References

Checked on 2026-09-10.

## Harness-Bench

https://github.com/Qihoo360/harness-bench

https://www.harness-bench.ai/

Current upstream documentation describes task-local `task.yaml`, prompt files, fixtures, and oracle grading, with adapters kept separate from the runner.

## ReAct

Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models”, ICLR 2023.

https://arxiv.org/abs/2210.03629

https://github.com/ysymyth/ReAct

## SWE-bench

https://github.com/SWE-bench/SWE-bench

Docker-based reproducible evaluation and executable grading are useful reference patterns for this repository.

## SWE-agent

https://github.com/SWE-agent/SWE-agent

The project documents a configurable, research-oriented agent interface and YAML-based configuration.

## AgentBench

https://github.com/THUDM/AgentBench

AgentBench separates environments and provides explicit agent/environment components for executable evaluation.

## OSWorld

https://github.com/xlang-ai/OSWorld

The evaluation flow separates environment setup, agent execution, post-processing, and execution-based evaluation.

## NanoBot

https://github.com/HKUDS/nanobot

The current documentation supports the one-shot command:

```text
nanobot agent -m "..."
```

## OpenClaw

https://github.com/openclaw/openclaw

The current CLI documentation supports:

```text
openclaw agent --agent main --message-file ./task.md --json
```

and documents `--local` for direct local execution.

## Additional benchmark references

## ToolBench

https://github.com/sambanova/toolbench

The repository provides executable evaluation infrastructure for software tool manipulation and reports execution success rates.

## tau-bench

https://github.com/superawind/tau-bench

The benchmark evaluates tool-agent-user interaction in realistic domains and provides trajectory-oriented evaluation infrastructure.
