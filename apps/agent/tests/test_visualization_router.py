import json

import httpx
import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import src.visualization_router as router
from src.visualization_router import (
    AUTO_DECISION,
    RouterError,
    decision_for,
    parse_decision,
    routed_request,
    routing_input,
)

ROUTER_ENV = [
    "ROUTER_BACKEND",
    "ROUTER_BASE_URL",
    "ROUTER_MODEL",
    "ROUTER_API_KEY",
    "ROUTER_EXTRA_BODY",
    "ROUTER_ON_ERROR",
    "ROUTER_MAX_TOKENS",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "ROUTER_TIMEOUT",
]


@pytest.fixture(autouse=True)
def clean_router_env(monkeypatch):
    for name in ROUTER_ENV:
        monkeypatch.delenv(name, raising=False)
    router._format_index.clear()
    yield
    router._format_index.clear()


def completion(content):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def decision_json(choice, confidence=0.9):
    return json.dumps({"visualization": choice, "confidence": confidence})


@pytest.mark.parametrize(
    ("choice", "renderer"),
    [
        ("table", "a2ui"),
        ("line_chart", "open_generative_ui"),
        ("map", "open_generative_ui"),
        ("animated_route", "open_generative_ui"),
        ("interactive_diagram", "open_generative_ui"),
        ("text", "text"),
    ],
)
def test_valid_decisions_map_to_exact_renderer(choice, renderer):
    decision = parse_decision(decision_json(choice))
    assert decision["renderer"] == renderer
    assert decision["visualization"] == choice
    assert decision["source"] == "router"


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        decision_json("invented"),
        '{"visualization": 3}',
        "I think a chart would be nice",
        "",
    ],
)
def test_bad_decisions_fail_loudly(content):
    with pytest.raises(RouterError, match="invalid visualization decision"):
        parse_decision(content)


@pytest.mark.parametrize(
    "content",
    [
        '<think>The user wants trends.</think>{"visualization": "line_chart", "confidence": 0.8}',
        '```json\n{"visualization": "line_chart", "confidence": 0.8}\n```',
        'Decision: {"visualization": "line_chart", "confidence": 0.8}.',
        '{"choice": "line_chart", "confidence": 0.8}',
        "line_chart",
        '"line_chart"',
    ],
)
def test_parser_tolerates_open_model_output_shapes(content):
    assert parse_decision(content)["visualization"] == "line_chart"


@pytest.mark.parametrize(("raw", "expected"), [(1.7, 1.0), (-2, 0.0), ("high", None), (True, None)])
def test_confidence_is_clamped_or_dropped(raw, expected):
    content = json.dumps({"visualization": "table", "confidence": raw})
    assert parse_decision(content)["confidence"] == expected


def test_continuation_reuses_decision_but_new_human_reroutes():
    state = {
        "messages": [HumanMessage(content="table", id="one")],
        "visualization_turn": "one",
        "visualization_decision": decision_for("table"),
    }
    assert routing_input(state) is None
    state["messages"].append(AIMessage(content="done"))
    assert routing_input(state) is None
    state["messages"].append(HumanMessage(content="now chart it", id="two"))
    turn, context = routing_input(state)
    assert turn == "two"
    assert context[-1]["content"] == "now chart it"


TOOL_NAMES = [
    "render_a2ui",
    "send_a2ui_json_to_client",
    "generate_a2ui",
    "generateSandboxedUi",
    "barChart",
    "pieChart",
    "query_data",
]


def request_for(decision):
    return ModelRequest(
        model=None,
        messages=[],
        tools=[{"name": n} for n in TOOL_NAMES],
        system_message=SystemMessage(content="base"),
        state={"visualization_decision": decision},
    )


@pytest.mark.parametrize(
    ("choice", "allowed"),
    [
        ("table", "send_a2ui_json_to_client"),
        ("line_chart", "generateSandboxedUi"),
        ("map", "generateSandboxedUi"),
        ("animated_route", "generateSandboxedUi"),
        ("text", None),
    ],
)
def test_route_filters_renderers_preserving_data_tools(choice, allowed):
    result = routed_request(request_for(decision_for(choice)))
    names = {t["name"] for t in result.tools}
    assert "query_data" in names
    assert ("generateSandboxedUi" in names) == (allowed == "generateSandboxedUi")
    assert ("send_a2ui_json_to_client" in names) == (
        allowed == "send_a2ui_json_to_client"
    )
    # @copilotkit/runtime's locked A2UI middleware injects render_a2ui.
    assert ("render_a2ui" in names) == (choice == "table")
    assert "barChart" not in names and "pieChart" not in names
    assert f"visualization={choice}" in result.system_message.content
    if choice == "map":
        assert "basemap.nationalmap.gov" in result.system_message.content
        assert "USGS attribution" in result.system_message.content
        assert "coverage limits" in result.system_message.content


def test_auto_decision_keeps_both_renderers_and_says_no_decision_was_made():
    result = routed_request(request_for(dict(AUTO_DECISION)))
    names = {t["name"] for t in result.tools}
    assert {"render_a2ui", "generateSandboxedUi", "query_data"} <= names
    assert "barChart" not in names and "pieChart" not in names
    assert "No routing decision was made" in result.system_message.content
    assert "copilotkit://open-generative-ui-tables" in result.system_message.content


def test_missing_endpoint_is_actionable():
    with pytest.raises(RouterError, match="ROUTER_BASE_URL"):
        router.route([])


def test_self_hosted_endpoint_needs_no_key_and_uses_compatible_payload(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm:8000/v1/")
    monkeypatch.setenv("LLM_MODEL", "qwen3-coder")
    monkeypatch.setenv("ROUTER_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}')
    seen = []

    def post(self, url, headers=None, json=None):
        seen.append((url, headers, json))
        return completion(decision_json("bar_chart"))

    monkeypatch.setattr(httpx.Client, "post", post)
    decision = router.route([{"role": "human", "content": "Compare sales by region"}])
    assert decision["visualization"] == "bar_chart"
    url, headers, payload = seen[0]
    assert url == "http://vllm:8000/v1/chat/completions"
    assert headers == {}
    assert payload["model"] == "qwen3-coder"
    assert payload["temperature"] == 0 and "max_tokens" in payload
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["visualization"]["enum"] == list(router.VISUALIZATIONS)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "Compare sales by region" in payload["messages"][1]["content"]


def test_router_specific_endpoint_overrides_answer_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://big:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "big-model")
    monkeypatch.setenv("ROUTER_BASE_URL", "http://small:11434/v1")
    monkeypatch.setenv("ROUTER_MODEL", "qwen3:4b")
    monkeypatch.setenv("ROUTER_API_KEY", "router-secret")
    base_url, model, key, trust_env, openai_cloud = router.endpoint()
    assert (base_url, model, key, openai_cloud) == ("http://small:11434/v1", "qwen3:4b", "router-secret", False)


def test_openai_cloud_payload_uses_completion_token_limit(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server-openai")
    seen = []

    def post(self, url, headers=None, json=None):
        seen.append((url, headers, json))
        return completion(decision_json("table"))

    monkeypatch.setattr(httpx.Client, "post", post)
    router.route([])
    url, headers, payload = seen[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers == {"Authorization": "Bearer server-openai"}
    assert payload["model"] == "chat-latest"
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload and "temperature" not in payload


def test_claude_answer_model_needs_explicit_router(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server-openai")
    monkeypatch.setenv("LLM_MODEL", "claude-opus-4-6")
    with pytest.raises(RouterError, match="ROUTER_MODEL"):
        router.endpoint()


@pytest.mark.parametrize("status", [400, 422])
def test_unsupported_response_formats_degrade_and_are_remembered(monkeypatch, status):
    monkeypatch.setenv("LLM_BASE_URL", "http://llama:8080/v1")
    monkeypatch.setenv("LLM_MODEL", "local")
    formats = []

    def post(self, url, headers=None, json=None):
        response_format = json.get("response_format")
        formats.append(response_format and response_format["type"])
        if response_format and response_format["type"] == "json_schema":
            return httpx.Response(status, text="response_format json_schema unsupported")
        return completion(decision_json("flowchart"))

    monkeypatch.setattr(httpx.Client, "post", post)
    assert router.route([])["visualization"] == "flowchart"
    assert router.route([])["visualization"] == "flowchart"
    assert formats == ["json_schema", "json_object", "json_object"]


def test_server_errors_degrade_one_request_without_downgrading_later_turns(monkeypatch):
    # llama-cpp-python answers an unsupported json_schema with 500, but a 500
    # can also be transient; it must not permanently skip json_schema.
    monkeypatch.setenv("LLM_BASE_URL", "http://llama:8080/v1")
    monkeypatch.setenv("LLM_MODEL", "local")
    formats = []
    statuses = iter([500, 200, 200])

    def post(self, url, headers=None, json=None):
        formats.append(json.get("response_format", {}).get("type"))
        status = next(statuses)
        return completion(decision_json("flowchart")) if status == 200 else httpx.Response(status)

    monkeypatch.setattr(httpx.Client, "post", post)
    assert router.route([])["visualization"] == "flowchart"
    assert router.route([])["visualization"] == "flowchart"
    assert formats == ["json_schema", "json_object", "json_schema"]


def test_servers_without_json_modes_use_plain_prompting(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://old:8080/v1")
    monkeypatch.setenv("LLM_MODEL", "local")
    formats = []

    def post(self, url, headers=None, json=None):
        formats.append(json.get("response_format", {}).get("type"))
        if "response_format" in json:
            return httpx.Response(400)
        return completion("calculator")

    monkeypatch.setattr(httpx.Client, "post", post)
    assert router.route([])["visualization"] == "calculator"
    assert formats == ["json_schema", "json_object", None]


@pytest.mark.parametrize("status", [401, 403, 404, 429])
def test_auth_model_and_rate_errors_do_not_retry_formats(monkeypatch, status):
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "local")
    calls = []

    def post(self, url, headers=None, json=None):
        calls.append(1)
        return httpx.Response(status)

    monkeypatch.setattr(httpx.Client, "post", post)
    with pytest.raises(RouterError, match=f"HTTP {status}"):
        router.route([])
    assert len(calls) == 1


def test_provider_errors_do_not_expose_response_body(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "local")
    monkeypatch.setattr(
        httpx.Client, "post", lambda *a, **k: httpx.Response(401, text="sensitive provider details")
    )
    with pytest.raises(RouterError, match="HTTP 401") as error:
        router.route([])
    assert "sensitive" not in str(error.value)


def test_timeout_is_actionable_without_secret_headers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-value")

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("private-test-value")

    monkeypatch.setattr(httpx.Client, "post", timeout)
    with pytest.raises(RouterError, match="timed out") as error:
        router.route([])
    assert "private-test-value" not in str(error.value)


def test_router_failure_falls_back_to_answer_model(monkeypatch):
    middleware = router.VisualizationRouterMiddleware()
    state = {"messages": [HumanMessage(content="Explain gears", id="one")]}
    update = middleware.before_agent(state, None)
    assert update == {"visualization_turn": "one", "visualization_decision": AUTO_DECISION}


def test_router_failure_can_fail_the_turn(monkeypatch):
    monkeypatch.setenv("ROUTER_ON_ERROR", "fail")
    middleware = router.VisualizationRouterMiddleware()
    state = {"messages": [HumanMessage(content="Explain gears", id="one")]}
    with pytest.raises(RouterError):
        middleware.before_agent(state, None)


def test_backend_none_skips_the_router_call(monkeypatch):
    monkeypatch.setenv("ROUTER_BACKEND", "none")
    monkeypatch.setattr(router, "route", lambda context: pytest.fail("router called"))
    middleware = router.VisualizationRouterMiddleware()
    state = {"messages": [HumanMessage(content="Explain gears", id="one")]}
    assert middleware.before_agent(state, None)["visualization_decision"] == AUTO_DECISION


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ROUTER_BACKEND", "jev"),
        ("ROUTER_ON_ERROR", "ignore"),
        ("ROUTER_TIMEOUT", "soon"),
        ("ROUTER_MAX_TOKENS", "0"),
        ("ROUTER_EXTRA_BODY", "[1]"),
    ],
)
def test_malformed_settings_fail_at_startup(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        router.VisualizationRouterMiddleware()


def test_separate_router_host_never_receives_the_answer_model_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.internal/v1")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("LLM_MODEL", "big")
    assert router.endpoint()[2] == "llm-secret"
    monkeypatch.setenv("ROUTER_BASE_URL", "http://router.elsewhere/v1")
    assert router.endpoint()[2] == ""
    monkeypatch.setenv("ROUTER_API_KEY", "router-secret")
    assert router.endpoint()[2] == "router-secret"


def test_server_key_follows_openai_base_url_but_byok_stays_on_openai(monkeypatch):
    from src.credentials import Credentials, current_credentials

    monkeypatch.setenv("OPENAI_API_KEY", "gateway-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.internal/v1/")
    assert router.endpoint()[:3] == ("https://gateway.internal/v1", "chat-latest", "gateway-key")
    token = current_credentials.set(Credentials("visitor"))
    try:
        assert router.endpoint()[:4] == ("https://api.openai.com/v1", "chat-latest", "visitor", False)
    finally:
        current_credentials.reset(token)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "", "reasoning_content": 'maybe {"visualization": "table"}'},
        {"role": "assistant", "content": None},
    ],
)
def test_empty_content_is_not_a_decision(monkeypatch, message):
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "local")
    monkeypatch.setattr(
        httpx.Client, "post", lambda *a, **k: httpx.Response(200, json={"choices": [{"message": message}]})
    )
    with pytest.raises(RouterError, match="ROUTER_MAX_TOKENS"):
        router.route([])


def test_unterminated_reasoning_is_not_a_decision():
    with pytest.raises(RouterError):
        parse_decision('<think>Not {"visualization": "table", "confidence": 1}; line_chart fits better')


def test_context_excludes_system_and_tool_content_and_is_bounded():
    from langchain_core.messages import ToolMessage

    state = {
        "messages": [
            SystemMessage(content="secret system"),
            ToolMessage(content="tool payload", tool_call_id="x"),
            HumanMessage(content="a" * 20000, id="bounded"),
        ]
    }
    _, context = routing_input(state)
    assert len(context) == 1
    assert len(context[0]["content"]) == 6000


def test_real_graph_filters_injected_frontend_tools_and_routes_once(monkeypatch):
    import asyncio
    from deepagents import create_deep_agent
    from copilotkit import CopilotKitMiddleware
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langgraph.checkpoint.memory import InMemorySaver

    bound = []
    calls = []

    class FakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            bound.append(
                {t.get("name") if isinstance(t, dict) else t.name for t in tools}
            )
            return self

    async def classify(context):
        calls.append(context)
        return decision_for("interactive_diagram")

    monkeypatch.setattr(router, "aroute", classify)
    graph = create_deep_agent(
        model=FakeModel(responses=[AIMessage(content="Ready")]),
        middleware=[CopilotKitMiddleware(), router.VisualizationRouterMiddleware()],
        checkpointer=InMemorySaver(),
    )

    async def run():
        config = {"configurable": {"thread_id": "routing-test"}}
        actions = [
            {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}},
            }
            for name in ["generateSandboxedUi", "render_a2ui", "send_a2ui_json_to_client", "barChart"]
        ]
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Explain gears", id="one")],
                "copilotkit": {"actions": actions},
            },
            config,
        )
        await graph.ainvoke({"messages": []}, config)
        assert len(calls) == 1
        await graph.ainvoke(
            {"messages": [HumanMessage(content="Now explain torque", id="two")]}, config
        )
        assert len(calls) == 2

    asyncio.run(run())
    assert bound and all(
        "generateSandboxedUi" in tools
        and "render_a2ui" not in tools
        and "barChart" not in tools
        and "send_a2ui_json_to_client" not in tools
        for tools in bound
    )


def test_idless_continuation_has_stable_turn_key():
    state = {"messages": [HumanMessage(content="table")]}
    turn, _ = routing_input(state)
    state.update(
        visualization_turn=turn,
        visualization_decision=decision_for("table"),
    )
    state["messages"].append(AIMessage(content="done"))
    assert routing_input(state) is None
    state["messages"].append(HumanMessage(content="table"))
    assert routing_input(state)[0] != turn
