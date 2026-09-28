# OpenClaw study proxy handoff

## What was breaking

The study built a temporary `ExperimentRunner` just to plan task cells. Runner
initialization rewrote the shared model dictionary with a local proxy URL. The
temporary runner then closed that proxy. The execution runner reused the
rewritten settings, so it created a second proxy pointed at the first, closed
proxy. This looked like a university endpoint or Docker networking outage.

Each runner now copies its agent/model settings before proxy setup. Matrix
planning uses a standalone function and does not start a proxy or open the run
database. The original served model identity remains part of every cell ID.
Proxy chain configuration is rejected. Native OpenClaw traffic passes through
the per-cell Docker sidecar, then the Windows host proxy, and only the host
proxy contacts the university endpoint. Trace names distinguish
`agent_sidecar_*` events from `host_relay_*` events; they describe the same
forwarded call at two points, not two independent model generations.

## Before starting the study

The prior launch and routed stores contain interrupted/failed attempts. They
are retained for audit and must not be reused under the corrected runner
identity. Use the separate connection-fix configuration and its fresh ignored
storage directory:

```powershell
.\.venv\Scripts\python.exe -m runner.cli openclaw-language-study --config configs/phase1.openclaw-language-connection-fix.yaml
```

This command performs the normal preflight, runs the English/Hindi OpenClaw
matrix sequentially, then runs the frozen judge gate and judges saved outputs.
It does not retry agent executions or judge calls. If service identity changes,
the request is rejected, or infrastructure fails, inspect the new store's
`journal.jsonl`, `heartbeat.json`, and `preflight-failure.json` before any
restart. Health events now include a redacted `failure_kind` and, for HTTP
responses, `status_code`; they never include request credentials or endpoint
URLs.

Read status without contacting the model:

```powershell
.\.venv\Scripts\python.exe -m runner.cli openclaw-language-status --config configs/phase1.openclaw-language-connection-fix.yaml
```

Review the first two task blocks in the trace directory before treating later
cells as operationally stable. In those traces, a sidecar event records what
OpenClaw sent, and the corresponding host-relay event records the forwarded
request and university response.

## Verification done for this fix

The regression tests construct two successive runners around a local fake
OpenAI endpoint, close the first proxy, verify the second still points to the
original endpoint, and count exactly one forwarded completion. Another test
proves matrix planning does not create proxies or a run database. No agent
study cells or university inference calls were launched for this repair.

The focused local suite passed 31 tests. One existing gate test was excluded
because it starts a Docker grader container and Docker Desktop was unavailable
in this session. The fake endpoint test proves runner ownership and forwarding;
the next operator should rely on the normal launch preflight to verify the live
Docker-to-host-to-university route before the first study cell.
