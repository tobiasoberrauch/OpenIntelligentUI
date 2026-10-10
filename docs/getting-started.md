# Getting started

Run Open Intelligent UI by CopilotKit locally with Node.js 22+, pnpm 9+, Python 3.12+, and [uv](https://docs.astral.sh/uv/).

## Install

```bash
git clone https://github.com/CopilotKit/OpenIntelligentUI.git
cd OpenIntelligentUI
make setup
```

`make setup` installs Node workspace dependencies and creates `apps/agent/.env` when absent. The agent's development command runs `uv sync` for Python dependencies.

## Configure the agent

Edit `apps/agent/.env`. For open models, point the agent at any OpenAI-compatible server; the visualization router uses the same endpoint unless you give it its own, or TypeSafe's Jev when `TYPESAFE_API_KEY` is set:

```dotenv
LLM_BASE_URL=http://localhost:11434/v1   # Ollama; vLLM: http://localhost:8000/v1
LLM_MODEL=qwen3-coder-32k                # the served model name
# Optional: a smaller, faster model just for routing
# ROUTER_MODEL=qwen2.5:3b
```

See [Open models](open-models.md) for vLLM, Ollama and llama.cpp setups, model requirements and tuning.

For hosted providers, remove `LLM_BASE_URL`: the model factory accepts `claude-*` names through Anthropic and `chat-latest` or `gpt-*` names through OpenAI. Set a model available to your account in `LLM_MODEL` and supply its provider key (`OPENAI_API_KEY` also serves routing). Unset `LLM_MODEL` uses the local default; an empty value or unsupported prefix fails with a configuration error.

For public demos, build the frontend with `NEXT_PUBLIC_BYOK=true`: visitors then test and save their own OpenAI key, and optionally a Jev key, under **API keys** in the chat header for the current browser session. Refreshing clears the keys.

Check model availability and access with a real request. This repository does not guarantee that every model name works or implement a separate native GPT-6 API. Never put shared provider keys in public frontend environment variables. See [BYOK behavior](../README.md#use-your-own-api-keys-optional).

## Configure the frontend

Optional settings belong in `apps/app/.env.local` or the frontend process environment:

| Variable                   | Default                 | Purpose                   |
| -------------------------- | ----------------------- | ------------------------- |
| `LANGGRAPH_DEPLOYMENT_URL` | `http://localhost:8123` | Agent URL                 |
| `MCP_SERVER_URL`           | Unset                   | Optional MCP integration  |
| `NEXT_PUBLIC_BYOK`         | Unset (off)             | Visitors bring an OpenAI key (and optionally a Jev key); read at build time |
| `RATE_LIMIT_ENABLED`       | `false`                 | Per-IP runtime rate limit |
| `RATE_LIMIT_WINDOW_MS`     | `60000`                 | Rate-limit window         |
| `RATE_LIMIT_MAX`           | `40`                    | Requests per window       |

The root [.env.example](../.env.example) documents both services' settings; placing variables only in a root `.env` is not the setup described here.

## Start and verify

```bash
make dev
```

This starts the frontend, agent, and MCP development processes. Open the [app](http://localhost:3000); the [agent](http://localhost:8123/health) and [frontend](http://localhost:3000/api/health) health endpoints should return `{"status":"ok"}`.

1. Open the chat interface and confirm the composer is usable. Rendering the chat shell does not require a successful model call.
2. Ask a simple factual question; the assistant should be able to answer directly in text.
3. Submit “Make a bill splitter with editable total, tip, and number of people.” Check that a generated tool works and rejects invalid input.
4. Ask a follow-up with changed assumptions. It should produce a new answer without requiring an earlier output to be patched.
5. Try a current-data request. Without a relevant source tool, the assistant should explain the limitation or use clearly labeled inputs, not invent live data.

A health response confirms process availability; only a real prompt tests provider authentication and generation. See [Verification and launch](deployment.md#verification-and-launch) for the broader checks.

```bash
make dev-app
make dev-agent
make dev-mcp
pnpm test
uv run --directory apps/agent pytest
make lint
make build
```

Run individual development commands in separate terminals when needed. See [Architecture](architecture.md) and [Product and agent behavior](interactive-answers.md).
