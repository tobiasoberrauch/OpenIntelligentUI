"""Model defaults and provider-specific validation."""

import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from langchain_anthropic import ChatAnthropic
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


def test_compatible_endpoint_accepts_any_open_model_name(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    model = build_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert model.openai_api_base == "http://localhost:8000/v1"
    assert model.openai_api_key.get_secret_value() == "EMPTY"
    assert model.use_responses_api is False
    # Unset so the server can use the remaining context for widget generation.
    assert model.max_tokens is None


def test_compatible_endpoint_options(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://litellm:4000/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-oss-120b")
    monkeypatch.setenv("LLM_API_KEY", "sk-litellm")
    monkeypatch.setenv("LLM_MAX_TOKENS", "32000")
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"top_k": 20}')
    model = build_model()
    assert model.openai_api_key.get_secret_value() == "sk-litellm"
    assert model.max_tokens == 32000
    assert model.extra_body == {"top_k": 20}


@pytest.mark.parametrize("raw", ["not json", "[1, 2]"])
def test_compatible_endpoint_rejects_bad_extra_body(monkeypatch, raw):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "local")
    monkeypatch.setenv("LLM_EXTRA_BODY", raw)
    with pytest.raises(ValueError, match="LLM_EXTRA_BODY"):
        build_model()


@pytest.mark.parametrize("value", [None, "  "])
def test_compatible_endpoint_requires_model_name(monkeypatch, value):
    # chat-latest must not be sent to a self-hosted server by default.
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    if value is None:
        monkeypatch.delenv("LLM_MODEL", raising=False)
    else:
        monkeypatch.setenv("LLM_MODEL", value)
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        build_model()


def test_credential_scoped_model_streams_each_chunk_once():
    # The wrapper reports every chunk it yields; the inner (tool-bound) model must
    # not report them again, or AG-UI streams duplicated text and tool-call deltas.
    import asyncio

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from src.model import CredentialScopedModel

    class ToolFake(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self.bind(tools=tools)

    inner = ToolFake(messages=iter([AIMessage(content="hello streaming world")]))
    tool = {"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}
    wrapped = CredentialScopedModel(fallback=inner).bind_tools([tool])

    async def run():
        names, text = [], []
        async for event in wrapped.astream_events("hi", version="v2"):
            if event["event"] == "on_chat_model_stream":
                names.append(event["name"])
                text.append(event["data"]["chunk"].content)
        return names, "".join(text)

    names, text = asyncio.run(run())
    assert names and set(names) == {"CredentialScopedModel"}
    assert text == "hello streaming world"
