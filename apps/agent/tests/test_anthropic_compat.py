"""CopilotKitMiddleware's "App Context" SystemMessage is appended after the
human turn by the add_messages reducer. Anthropic requires system messages at
the head; vLLM/SGLang (via LiteLLM) reject more than ONE system message even
when consecutive ("System message must be at the beginning"). The middleware
therefore merges all system content into a single leading message."""

import asyncio
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.anthropic_compat import ConsecutiveSystemMessagesMiddleware
from src.model import build_model


def _wrap(messages):
    middleware = ConsecutiveSystemMessagesMiddleware()
    request = ModelRequest(model=build_model(), messages=list(messages))
    seen = {}

    def handler(req):
        seen["messages"] = list(req.messages)
        return "called"

    result = middleware.wrap_model_call(request, handler)
    assert result == "called"
    return seen["messages"]


def test_moves_trailing_system_message_to_front():
    human = HumanMessage(content="build an airplane")
    ctx = SystemMessage(content="App Context: design skill")
    assert _wrap([human, ctx]) == [ctx, human]


def test_merges_system_messages_into_single_leading_message():
    s1 = SystemMessage(content="first")
    human = HumanMessage(content="hi")
    s2 = SystemMessage(content="App Context: second")
    result = _wrap([s1, human, s2])
    assert len(result) == 2
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "first\n\nApp Context: second"
    assert result[1] == human


def test_merges_into_request_system_message_when_present():
    middleware = ConsecutiveSystemMessagesMiddleware()
    main = SystemMessage(content="main prompt")
    human = HumanMessage(content="hi")
    ctx = SystemMessage(content="App Context: design skill")
    request = ModelRequest(model=build_model(), messages=[human, ctx], system_message=main)
    seen = {}

    def handler(req):
        seen["messages"] = list(req.messages)
        seen["system"] = req.system_message
        return "called"

    middleware.wrap_model_call(request, handler)
    assert seen["messages"] == [human]
    assert seen["system"].content == "main prompt\n\nApp Context: design skill"


def test_single_system_message_total_for_vllm_backends():
    # vLLM via LiteLLM 400s on >1 system message anywhere in the payload.
    middleware = ConsecutiveSystemMessagesMiddleware()
    main = SystemMessage(content="main")
    request = ModelRequest(
        model=build_model(),
        messages=[HumanMessage(content="hi"), SystemMessage(content="ctx"), SystemMessage(content="ctx2")],
        system_message=main,
    )
    seen = {}

    def handler(req):
        seen["messages"] = list(req.messages)
        seen["system"] = req.system_message
        return "called"

    middleware.wrap_model_call(request, handler)
    total = sum(1 for m in seen["messages"] if getattr(m, "type", None) in ("system", "developer"))
    if seen["system"] is not None:
        total += 1
    assert total == 1


def test_noop_when_already_consecutive():
    s1 = SystemMessage(content="first")
    human = HumanMessage(content="hi")
    ai = AIMessage(content="ok")
    assert _wrap([s1, human, ai]) == [s1, human, ai]


def test_async_path_reorders_too():
    middleware = ConsecutiveSystemMessagesMiddleware()
    human = HumanMessage(content="hi")
    ctx = SystemMessage(content="App Context: x")
    request = ModelRequest(model=build_model(), messages=[human, ctx])
    seen = {}

    async def handler(req):
        seen["messages"] = list(req.messages)
        return "called"

    result = asyncio.run(middleware.awrap_model_call(request, handler))
    assert result == "called"
    assert seen["messages"] == [ctx, human]


def test_repairs_thinking_block_missing_text_field():
    # Streaming a zero-length thinking block yields {type, signature} with no
    # "thinking" key; Anthropic 400s on replay unless the field is restored.
    broken = AIMessage(
        content=[
            {"type": "thinking", "signature": "sig123"},
            {"type": "text", "text": "I'll build a plane."},
        ]
    )
    human = HumanMessage(content="build it")
    messages = _wrap([human, broken])
    repaired_block = messages[-1].content[0]
    assert repaired_block["thinking"] == ""
    assert repaired_block["signature"] == "sig123"
    assert messages[-1].content[1] == {"type": "text", "text": "I'll build a plane."}


def test_intact_thinking_block_unchanged():
    ok = AIMessage(
        content=[{"type": "thinking", "thinking": "hmm", "signature": "s"}]
    )
    messages = _wrap([HumanMessage(content="hi"), ok])
    assert messages[-1].content[0]["thinking"] == "hmm"


def test_agent_registers_compat_middleware():
    from pathlib import Path

    main_src = (Path(__file__).parent.parent / "main.py").read_text()
    assert "ConsecutiveSystemMessagesMiddleware" in main_src
