# Visualization routing

Before each new user message, a router makes one typed presentation decision: which of thirteen visualizations fits, and therefore which renderer the answer model may use. The router is a structured-output request (`apps/agent/src/visualization_router.py`) to any OpenAI-compatible chat-completions endpoint: vLLM, SGLang, Ollama, llama.cpp `llama-server`, LiteLLM, LocalAI, TGI or OpenAI. No proprietary routing service is involved.

The router receives up to four recent user/assistant text messages, capped at 6,000 characters each, as delimited data. It does not receive API keys, system instructions, raw tool outputs, or generated tool code. Its answer is constrained to `{"visualization": <enum>, "confidence": <number>}`.

| Selected visualization | Renderer |
| --- | --- |
| Text | Normal chat answer |
| Table | A2UI Table catalog |
| Bar, line, scatter, distribution, part-to-whole chart | Open Generative UI |
| Flowchart, static or interactive diagram | Open Generative UI |
| Calculator | Open Generative UI |
| Map or animated itinerary | Open Generative UI |

The renderer is derived from the router's validated choice, preventing contradictory renderer/type pairs. The selection is stored with the user message ID in thread state and reused during tool continuations. A new user message is classified again with conversation context. Middleware filters rendering tools after CopilotKit injects them; unrelated data/state tools remain available. The answer model receives the selected visualization type plus the relevant rendering contract.

The A2UI catalog supplies a semantic HTML table with column headers, a caption, keyboard-focusable horizontal scrolling, and a required provenance label. It accepts at most 12 columns and 200 rows and rejects mismatched row widths. React renders values as text. Charts and diagrams continue to use the existing isolated sandbox.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ROUTER_BACKEND` | `llm` | `llm` routes with a model; `none` skips routing and lets the answer model choose |
| `ROUTER_BASE_URL` | `LLM_BASE_URL`, else OpenAI | OpenAI-compatible endpoint for routing |
| `ROUTER_MODEL` | `LLM_MODEL` | Served model name; a small instruct model is enough |
| `ROUTER_API_KEY` | `LLM_API_KEY`, else `OPENAI_API_KEY` | Sent as a bearer token only when set |
| `ROUTER_EXTRA_BODY` | Unset | JSON merged into the request, e.g. `{"chat_template_kwargs": {"enable_thinking": false}}` for Qwen3 on vLLM |
| `ROUTER_MAX_TOKENS` | `512` | Output budget; raise it for reasoning models that think before answering |
| `ROUTER_TIMEOUT` | `30` | Seconds per request |
| `ROUTER_ON_ERROR` | `auto` | `auto`: on failure the answer model chooses; `fail`: fail the turn with a retryable error |

Servers differ in structured-output support. The router asks for `json_schema` first, then `json_object`, then plain prompting; whichever format the endpoint accepts is remembered per endpoint and model. The parser tolerates `<think>` blocks, code fences and prose around the JSON and accepts a bare option name. Unknown options are rejected.

A routing model only classifies, so it can be much smaller and faster than the answer model: a 3–8B instruct model on the same GPU, or a separate Ollama instance, keeps the added latency low. Disable thinking for hybrid reasoning models (`ROUTER_EXTRA_BODY`), or the routing call spends its budget on reasoning.

When routing fails or is turned off, the stored decision says `source: "answer_model"`; the answer model keeps both table and sandbox renderers and is told that no routing decision was made. Nothing claims the router decided. Authentication, missing-model, rate-limit and timeout errors are logged and never expose provider response bodies. Successful health checks confirm the server is running, not model access.

Routing is a model judgment, not a guarantee of the objectively best visualization. Missing source data still requires clarification or explicit illustrative assumptions. Live provider checks and browser checks complement the deterministic routing, error-path, continuation, and A2UI rendering tests.

## Maps and animated itineraries

Maps use Leaflet with live USGS topographic tiles. For trips, `get_trip_stop_images` retrieves Wikipedia destination photos with validated Wikimedia Commons attribution; unavailable photos remain text cards. These images are sourced photographs, not live camera feeds.

The sandbox and downloaded HTML include `createTripAnimator`. Its default six-second sequence drops numbered pins into place one at a time, reveals connections behind them, and synchronizes destination cards. Pause/resume, replay, manual stop selection, and reduced-motion behavior share the same controller. The map camera stays fixed during playback, and the chat does not auto-scroll to follow the animation.

Basemap tiles do not establish driving routes, traffic, or road closures. Connecting lines are explicitly illustrative unless backed by directions data; the agent must not invent verified mileage or driving times.
