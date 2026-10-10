from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage

from src.openai_compat import SingleSystemMessageMiddleware, _merged


def test_app_context_system_messages_fold_into_one_system_prompt():
    request = ModelRequest(
        model=None,
        messages=[
            SystemMessage(content="App Context: design skill"),
            HumanMessage(content="hi"),
            SystemMessage(content=[{"type": "text", "text": "late note"}]),
        ],
        tools=[],
        system_message=SystemMessage(content="base prompt"),
        state={},
    )
    result = _merged(request)
    assert [m.type for m in result.messages] == ["human"]
    assert result.system_message.content == "base prompt\n\nApp Context: design skill\n\nlate note"


def test_requests_without_extra_system_messages_are_unchanged():
    request = ModelRequest(
        model=None,
        messages=[HumanMessage(content="hi")],
        tools=[],
        system_message=SystemMessage(content="base"),
        state={},
    )
    assert _merged(request) is request


def test_middleware_passes_merged_request_to_handler():
    seen = []
    request = ModelRequest(
        model=None,
        messages=[SystemMessage(content="ctx"), HumanMessage(content="hi")],
        tools=[],
        system_message=None,
        state={},
    )
    SingleSystemMessageMiddleware().wrap_model_call(request, seen.append)
    assert seen[0].system_message.content == "ctx"
    assert len(seen[0].messages) == 1
