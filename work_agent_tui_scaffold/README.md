# work-agent

A lightweight, pure-Python, WSL-friendly agentic harness for repo inspection, eval repair, shell execution, patching, read-only SQL, and optional MCP/local-model tools.

The core is headless:

```text
model -> structured AgentAction -> Python ToolRegistry -> observation -> next action
```

The TUI is only a presentation and approval layer around the same driver.

## Install

```bash
cd work_agent_tui_scaffold
python -m venv .venv
source .venv/bin/activate
pip install -e '.[tui]'
```

Minimum non-TUI install:

```bash
pip install -e .
```

## Azure OpenAI env

```bash
export AZURE_OPENAI_BASE_URL='https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/'
export AZURE_OPENAI_API_KEY='...'
export AZURE_OPENAI_DEPLOYMENT='your-gpt-5-2-deployment-name'
```

For local smoke testing without a model call:

```bash
export WORK_AGENT_MOCK=1
```

## CLI examples

```bash
work-agent run \
  --repo /work/forms-agent \
  --task "List the repo root, inspect README if present, run git status, and finish." \
  --policy configs/policy.yaml \
  --mode diagnose
```

TUI:

```bash
work-agent tui \
  --repo /work/forms-agent \
  --task "Run git status, inspect README if present, and finish." \
  --policy configs/policy.yaml \
  --mode diagnose
```

Forms eval repair:

```bash
work-agent repair \
  --repo /work/forms-agent \
  --errors /work/evals/errors.csv \
  --profile configs/profiles/forms_agent.yaml \
  --policy configs/policy.yaml \
  --mode patch-assist
```

TUI eval repair:

```bash
work-agent tui-repair \
  --repo /work/forms-agent \
  --errors /work/evals/errors.csv \
  --profile configs/profiles/forms_agent.yaml \
  --policy configs/policy.yaml \
  --mode patch-assist
```

## TUI keys

```text
q / Ctrl-C    quit UI, request agent stop
s             request stop without immediately exiting
 a            approve pending action
 r            reject pending action
 d            toggle detail pane focus text
 ?            toggle help
```

## Design notes

- No npm, no node, no Cline.
- Shell execution uses `subprocess.run(..., shell=False)`.
- Shell is allowlisted and timed out.
- Patches must be unified git diffs.
- SQL is read-only by policy.
- MCP is an optional adapter.
- The TUI is optional; CLI and batch modes use the same driver.
