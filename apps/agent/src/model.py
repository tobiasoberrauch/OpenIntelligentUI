"""Chat model factory for the agent."""

import os
from contextlib import contextmanager

from langchain.agents.middleware import AgentMiddleware
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI
from pydantic import Field
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_MODEL = "chat-latest"

# langchain-anthropic defaults max_tokens to 4096, which truncates
# generateSandboxedUi tool args mid-stream (jsFunctions/jsExpressions never
# arrive and the widget is stuck in its preview sandbox). Widget generation
# routinely needs tens of thousands of output tokens.
MAX_TOKENS = 64000


class DedupChatOpenAI(ChatOpenAI):
    """ChatOpenAI that drops duplicate tool-call headers some vLLM backends
    re-emit mid-stream (id+name, with or without args).

    Filtering here matters: the raw callback events that AG-UI consumers
    convert are emitted from these chunks, before any wrapper sees the
    aggregated output. A strict consumer rejects the duplicate
    TOOL_CALL_START ("already in progress") and aborts the whole run.
    """

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        seen_headers = set()
        for chunk in super()._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            if _is_duplicate_tool_header(seen_headers, chunk):
                continue
            yield chunk

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        seen_headers = set()
        async for chunk in super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            if _is_duplicate_tool_header(seen_headers, chunk):
                continue
            yield chunk


def build_model(*, allow_unconfigured: bool = False) -> BaseChatModel:
    model_name = os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    # With OPENAI_BASE_URL set, any model name routes to that OpenAI-compatible
    # endpoint (e.g. a LiteLLM router); otherwise only built-in prefixes apply.
    is_openai = bool(base_url) or model_name == "chat-latest" or model_name.startswith("gpt-")
    if not is_openai and not model_name.startswith("claude-"):
        raise ValueError(
            "LLM_MODEL must name a supported provider: claude-* (Anthropic), "
            "gpt-*/chat-latest (OpenAI), or any model with OPENAI_BASE_URL set. "
            "Unset LLM_MODEL to use the default."
        )
    key_name = "OPENAI_API_KEY" if is_openai else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_name, "").strip():
        if allow_unconfigured:
            return UnconfiguredModel()
        raise ValueError(
            f"{key_name} is required for the selected LLM_MODEL provider. "
            "Set it in the agent environment and restart the agent."
        )
    if is_openai:
        # chat-latest follows ChatGPT Instant; explicit gpt-* pins also work.
        temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
        if base_url:
            # Reasoning models behind a router need a large output budget;
            # widget generation truncates below ~32k tokens.
            return DedupChatOpenAI(
                model=model_name,
                base_url=base_url,
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", MAX_TOKENS)),
                temperature=temperature,
            )
        return DedupChatOpenAI(model=model_name, temperature=temperature)
    return ChatAnthropic(
        model=model_name,
        max_tokens=MAX_TOKENS,
    )


class UnconfiguredModel(BaseChatModel):
    """Compile a BYOK-only graph without constructing a provider with fake credentials."""

    @property
    def _llm_type(self):
        return "unconfigured"

    def _generate(self, *args, **kwargs):
        raise ValueError("Provider credentials are required. Add OpenAI and Jev keys.")

    def bind_tools(self, tools, **kwargs):
        return self


def request_model(fallback):
    from src.credentials import current_credentials
    credentials = current_credentials.get()
    if credentials is None:
        return fallback
    from langchain_openai import ChatOpenAI
    from openai import AsyncOpenAI, OpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient

    # SDK None defaults inherit server organization/project settings. Explicit
    # clients with empty scopes and isolated transports prevent that inheritance,
    # including server HTTP(S)_PROXY and OPENAI_PROXY settings.
    options = {
        "api_key": credentials.openai,
        "base_url": "https://api.openai.com/v1",
        "organization": "",
        "project": "",
    }
    sync_client = OpenAI(**options, http_client=DefaultHttpxClient(trust_env=False))
    async_client = AsyncOpenAI(
        **options, http_client=DefaultAsyncHttpxClient(trust_env=False)
    )
    # After construction, None omits these headers without rereading env vars.
    for client in (sync_client, async_client):
        client.organization = None
        client.project = None
        client.admin_api_key = None
        client.webhook_secret = None
        # The SDK merges OPENAI_CUSTOM_HEADERS even with default_headers={}.
        # Clear that merge before any request so host Authorization cannot win.
        client._custom_headers = {}
    model = ChatOpenAI(
        model=DEFAULT_MODEL,
        api_key=credentials.openai,
        base_url=options["base_url"],
        openai_proxy="",
        root_client=sync_client,
        client=sync_client.chat.completions,
        root_async_client=async_client,
        async_client=async_client.chat.completions,
    )
    # LangChain's metadata field separately inherits organization environment
    # values even when prebuilt SDK clients are supplied. Clear the inert field.
    model.openai_organization = None
    return model


@contextmanager
def provider_error_boundary():
    from src.credentials import current_credentials
    try:
        yield
    except Exception:
        if current_credentials.get() is None:
            raise
        raise ValueError("OpenAI request failed. Check your key and model access, or retry.") from None


@contextmanager
def _suppress_inner_events():
    # This wrapper streams its target model, so both layers emit
    # on_chat_model_stream events for the same chunks. AG-UI converters
    # process both and see every tool header twice, then reject the
    # duplicate TOOL_CALL_START ("already in progress") and abort the run.
    # Clearing the ambient runnable config for the inner call suppresses its
    # events; only this layer streams to the consumer.
    from langchain_core.runnables.config import var_child_runnable_config

    token = var_child_runnable_config.set({})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


class RequestModelMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        with provider_error_boundary():
            return handler(request.override(model=request_model(request.model)))

    async def awrap_model_call(self, request, handler):
        with provider_error_boundary():
            return await handler(request.override(model=request_model(request.model)))


class CredentialScopedModel(BaseChatModel):
    """Route DeepAgents' internal summary/subagent model calls through BYOK too."""

    fallback: BaseChatModel = Field(exclude=True)

    @property
    def _llm_type(self):
        return "credential-scoped"

    def bind_tools(self, tools, **kwargs):
        return self.bind(_credential_tools=tools, _credential_tool_options=kwargs)

    def _target(self, kwargs):
        model = request_model(self.fallback)
        options = dict(kwargs)
        tools = options.pop("_credential_tools", None)
        tool_options = options.pop("_credential_tool_options", {})
        if tools is not None:
            model = model.bind_tools(tools, **tool_options)
        return model, options

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        with provider_error_boundary():
            model, options = self._target(kwargs)
            with _suppress_inner_events():
                message = model.invoke(messages, stop=stop, **options)
            return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        with provider_error_boundary():
            model, options = self._target(kwargs)
            with _suppress_inner_events():
                message = await model.ainvoke(messages, stop=stop, **options)
            return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        with provider_error_boundary():
            model, options = self._target(kwargs)
            seen_headers = set()
            with _suppress_inner_events():
                for chunk in model.stream(messages, stop=stop, **options):
                    message = as_chunk(chunk)
                    if _is_duplicate_tool_header(seen_headers, message):
                        continue
                    yield ChatGenerationChunk(message=message)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        with provider_error_boundary():
            model, options = self._target(kwargs)
            seen_headers = set()
            with _suppress_inner_events():
                async for chunk in model.astream(messages, stop=stop, **options):
                    message = as_chunk(chunk)
                    if _is_duplicate_tool_header(seen_headers, message):
                        continue
                    yield ChatGenerationChunk(message=message)


def _is_duplicate_tool_header(seen, chunk) -> bool:
    # Some vLLM backends re-emit a complete tool-call header (id+name, with or
    # without args) after already sending it. Strict AG-UI consumers reject
    # the duplicate TOOL_CALL_START ("already in progress") and abort the run.
    # Only header entries carry id+name; arg fragments (id=None) are never
    # treated as duplicates. Accepts both AIMessageChunk and
    # ChatGenerationChunk (which wraps the message in .message).
    message = getattr(chunk, "message", chunk)
    tool_call_chunks = getattr(message, "tool_call_chunks", None) or []
    if not tool_call_chunks:
        return False
    keys = []
    for entry in tool_call_chunks:
        if not (entry.get("id") and entry.get("name")):
            continue
        key = (
            entry.get("index"),
            entry.get("id"),
            entry.get("name"),
            entry.get("args") or "",
        )
        if key in seen:
            return True
        keys.append(key)
    seen.update(keys)
    return False


def as_chunk(message):
    if isinstance(message, AIMessageChunk):
        return message
    return AIMessageChunk(**message.model_dump(exclude={"type"}))
