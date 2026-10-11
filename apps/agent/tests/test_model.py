"""Model defaults and provider-specific validation."""

import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI

from src.model import build_model


def test_default_model_is_chatgpt_instant(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    model = build_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "chat-latest"


def test_llm_model_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_MODEL", "claude-opus-4-6")
    model = build_model()
    assert model.model == "claude-opus-4-6"


def test_gpt_model_names_route_to_openai_fallback(monkeypatch: pytest.MonkeyPatch):
    # Production fallback: flipping LLM_MODEL to a gpt-* name in the Render
    # dashboard must work without a code change.
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4-2026-03-05")
    model = build_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-5.4-2026-03-05"


def test_claude_model_names_route_to_anthropic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_MODEL", "claude-opus-4-6")
    assert isinstance(build_model(), ChatAnthropic)


def test_max_tokens_fits_full_widget_generation(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "claude-fable-5")
    # langchain-anthropic's default (4096) truncates generateSandboxedUi args
    # mid-stream: css+html arrive but jsFunctions/jsExpressions are cut off, so
    # htmlComplete never fires and the widget never leaves the preview sandbox.
    model = build_model()
    assert model.max_tokens >= 32000


def test_openai_base_url_routes_custom_models(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen3.8-27b")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://10.200.3.60:4000/v1")
    monkeypatch.setenv("LLM_MAX_TOKENS", "32000")
    model = build_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "qwen3.8-27b"
    assert str(model.openai_api_base).rstrip("/") == "http://10.200.3.60:4000/v1"
    assert model.max_tokens == 32000


def test_custom_endpoint_defaults_to_large_max_tokens(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    monkeypatch.setenv("LLM_MODEL", "audius-fast")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://10.200.3.60:4000/v1")
    model = build_model()
    assert isinstance(model, ChatOpenAI)
    assert (model.max_tokens or 0) >= 32000


def test_custom_model_without_base_url_is_rejected(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "qwen3.8-27b")
    with pytest.raises(ValueError, match="LLM_MODEL"):
        build_model()


class _DuplicateHeaderModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "duplicate-header-fake"

    def _generate(self, *args, **kwargs):
        raise NotImplementedError

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import ToolCallChunk
        from langchain_core.outputs import ChatGenerationChunk

        empty_header = ToolCallChunk(index=0, id="call_x", name="render_a2ui", args="", type="tool_call_chunk")
        full_header = ToolCallChunk(index=0, id="call_y", name="render_a2ui", args='{"a": 1}', type="tool_call_chunk")
        args_piece = ToolCallChunk(index=0, id=None, name=None, args='{"a":', type="tool_call_chunk")
        yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[empty_header]))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[empty_header]))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[full_header]))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[full_header]))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[args_piece]))
        yield ChatGenerationChunk(message=AIMessageChunk(content="done"))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self._stream(messages, stop=stop):
            yield chunk


def test_duplicate_empty_tool_headers_are_dropped(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "chat-latest")
    from src.model import CredentialScopedModel

    wrapped = CredentialScopedModel(fallback=_DuplicateHeaderModel())
    messages = list(wrapped.stream([HumanMessage(content="hi")]))
    headers = [
        c
        for message in messages
        for c in (getattr(message, "tool_call_chunks", None) or [])
        if c.get("id") in ("call_x", "call_y")
    ]
    assert len(headers) == 2
    assert any(c.get("id") == "call_x" for c in headers)
    assert any(c.get("id") == "call_y" for c in headers)
    assert any(getattr(message, "content", "") == "done" for message in messages)


def test_duplicate_empty_tool_headers_dropped_async(monkeypatch):
    import asyncio

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "chat-latest")
    from src.model import CredentialScopedModel

    wrapped = CredentialScopedModel(fallback=_DuplicateHeaderModel())

    async def collect():
        collected = []
        async for message in wrapped.astream([HumanMessage(content="hi")]):
            collected.append(message)
        return collected

    messages = asyncio.run(collect())
    headers = [
        c
        for message in messages
        for c in (getattr(message, "tool_call_chunks", None) or [])
        if c.get("id") in ("call_x", "call_y")
    ]
    assert len(headers) == 2


def test_dedup_chat_openai_filters_duplicate_headers(monkeypatch):
    import asyncio

    from langchain_core.messages import ToolCallChunk
    from langchain_core.outputs import ChatGenerationChunk
    from src.model import DedupChatOpenAI

    header = ToolCallChunk(index=0, id="call_z", name="render_a2ui", args='{"a": 1}', type="tool_call_chunk")

    async def fake_astream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[header]))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[header]))

    monkeypatch.setattr(ChatOpenAI, "_astream", fake_astream)
    model = DedupChatOpenAI(model="audius-fast", api_key="test-key")

    async def collect():
        return [chunk async for chunk in model.astream([HumanMessage(content="hi")])]

    messages = asyncio.run(collect())
    headers = [
        c
        for message in messages
        for c in (getattr(message, "tool_call_chunks", None) or [])
        if c.get("id") == "call_z"
    ]
    assert len(headers) == 1


def test_main_uses_build_model():
    import ast
    from pathlib import Path

    main_src = Path(__file__).parent.parent / "main.py"
    tree = ast.parse(main_src.read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "build_model" in names
    assert "ChatOpenAI" not in main_src.read_text()


@pytest.mark.parametrize("name", ["", "   ", "gemini-example", "typo-model"])
def test_invalid_model_configuration_fails_before_client_creation(monkeypatch, name):
    monkeypatch.setenv("LLM_MODEL", name)
    with pytest.raises(ValueError, match="LLM_MODEL"):
        build_model()


@pytest.mark.parametrize(
    ("model", "key"),
    [
        ("claude-opus-4-6", "ANTHROPIC_API_KEY"),
        ("gpt-5.4-2026-03-05", "OPENAI_API_KEY"),
        ("chat-latest", "OPENAI_API_KEY"),
    ],
)
@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_provider_key_has_actionable_error(monkeypatch, model, key, value):
    monkeypatch.setenv("LLM_MODEL", model)
    if value is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        build_model()
