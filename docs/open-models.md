# Open models

Open Intelligent UI runs entirely on open components: CopilotKit, AG-UI, LangGraph/Deep Agents and A2UI are open source, and both model roles (the **answer model** and the **visualization router**) can talk to any OpenAI-compatible chat-completions server. Nothing requires a proprietary API; TypeSafe's Jev remains available as a routing backend (`ROUTER_BACKEND=jev`, or `auto` with `TYPESAFE_API_KEY`). To keep routing on your own infrastructure, leave `TYPESAFE_API_KEY` unset or set `ROUTER_BACKEND=llm`.

| Role | Variables | What it needs |
| --- | --- | --- |
| Answer model | `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | Reliable tool calling with streamed arguments, long outputs, 32k+ context |
| Visualization router | `ROUTER_BASE_URL`, `ROUTER_MODEL`, `ROUTER_API_KEY` (default: the answer model's) | Instruction following and JSON output; a small model is enough |

## Choosing models

The answer model drives a tool-heavy agent. `generateSandboxedUi` streams HTML, CSS and JavaScript as tool arguments in a fixed order and often produces 5,000–20,000 output tokens; the system prompt, skills and tool schemas need a context of at least 32k tokens. Text answers and A2UI tables work with smaller models; interactive widgets, maps and charts need strong coding and tool-calling ability. Good candidates are recent open-weight coding/agent models with permissive licenses, for example the Qwen3-Coder and Qwen3 families (Apache-2.0), gpt-oss (Apache-2.0), GLM-4.x (MIT) or Mistral's Devstral/Mistral Small (Apache-2.0). Check current versions, licenses and your server's tool-call parser before choosing; model quality matters more here than in a plain chat app.

The router only classifies each user turn into one of thirteen presentations. A 3–8B instruct model is usually enough; in testing, even a 1.5B model on CPU picked the expected renderer for typical requests. Turn off thinking for hybrid reasoning models (`ROUTER_EXTRA_BODY`), or the call spends its budget on reasoning.

## vLLM (GPU servers, production)

Tool calling needs `--enable-auto-tool-choice` and the parser matching the model family (see the [vLLM tool-calling docs](https://docs.vllm.ai/en/latest/features/tool_calling.html)):

```bash
# Answer model
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --served-model-name qwen3-coder \
  --max-model-len 65536 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --port 8000

# Optional: small routing model on a second port (no tool calling needed)
vllm serve Qwen/Qwen3-4B-Instruct-2507 --served-model-name router \
  --max-model-len 8192 --gpu-memory-utilization 0.15 --port 8001
```

```dotenv
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=qwen3-coder
ROUTER_BASE_URL=http://localhost:8001/v1
ROUTER_MODEL=router
```

For gpt-oss use `--tool-call-parser openai`; for GLM-4.5/4.6 `glm45`. vLLM supports `json_schema` structured outputs, so routing decisions are constrained to the allowed options. Leave `LLM_MAX_TOKENS` unset: vLLM rejects a `max_tokens` larger than the remaining context, while unset lets each widget use the full remaining budget.

## Ollama (workstations, development)

Ollama's OpenAI-compatible API cannot set the context size per request, and the default context is too small for this agent: Ollama truncates the prompt instead of failing. Create a model variant with a larger `num_ctx` ([Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)):

```bash
ollama pull qwen3-coder:30b
printf 'FROM qwen3-coder:30b\nPARAMETER num_ctx 32768\n' > Modelfile
ollama create qwen3-coder-32k -f Modelfile
ollama pull qwen2.5:3b   # optional, small non-thinking router model
```

```dotenv
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3-coder-32k
ROUTER_MODEL=qwen2.5:3b
```

If you route with a thinking model instead, raise `ROUTER_MAX_TOKENS` so the answer fits after the reasoning.

## llama.cpp `llama-server`

`--jinja` enables the model's chat template, which tool calling requires:

```bash
llama-server -m Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --jinja -c 32768 --port 8080
```

```dotenv
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=qwen3-coder   # llama-server serves one model; the name is informational
```

## LiteLLM and gateways

A LiteLLM proxy or another gateway can front several backends (vLLM clusters, Azure OpenAI, Bedrock) behind one OpenAI-compatible URL. Set `LLM_API_KEY` (and `ROUTER_API_KEY` if the router uses a different key) to the gateway key.

## What the agent adapts automatically

- With `LLM_BASE_URL` set, every system message (including CopilotKit's injected app context) is merged into one leading system prompt, because many open-model chat templates accept only one.
- The answer model uses Chat Completions, not OpenAI's Responses API.
- The router negotiates the structured-output format (`json_schema`, `json_object`, plain text) and remembers what the endpoint supports.
- `LLM_EXTRA_BODY` and `ROUTER_EXTRA_BODY` pass server-specific options such as `{"chat_template_kwargs": {"enable_thinking": false}}` (Qwen3 on vLLM) or `{"top_k": 20}`.

The agent's first request carries roughly 7,000–8,000 prompt tokens (system prompt, skills, tool schemas). On a GPU that prefill takes well under a second; on CPU-only or heavily shared hardware it can exceed langchain-openai's default 120-second wait for the first streamed chunk. Raise or disable it with `LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` (`0` disables). Servers with prefix caching (vLLM, llama.cpp) reuse the shared prompt prefix on later turns.

## Network access beyond the model

Local models keep prompts and answers on your infrastructure, but generated answers can still load public resources in the browser: chart and map libraries from public CDNs (jsDelivr, unpkg, cdnjs), USGS map tiles, and Wikipedia/Wikimedia photos for trip itineraries (fetched by the agent's `get_trip_stop_images` tool). For an air-gapped deployment, mirror those libraries and adjust the skills in `apps/agent/skills/` and the design system, or restrict the sandbox's allowed hosts.

## Verify

```bash
# Router decision from your endpoint
curl -s $LLM_BASE_URL/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'"$LLM_MODEL"'",
  "messages": [{"role": "user", "content": "Reply with JSON {\"visualization\": \"table\", \"confidence\": 1}"}],
  "response_format": {"type": "json_object"}, "max_tokens": 50}'

uv run --directory apps/agent pytest
make dev
```

Then follow the checks in [Getting started](getting-started.md#start-and-verify). A text answer proves connectivity; only a generated chart, calculator or map proves that the answer model handles the streaming tool contract.
