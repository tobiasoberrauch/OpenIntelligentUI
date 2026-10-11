"""Provider message normalization, applied at model-call time so graph state
stays untouched.

1. CopilotKitMiddleware injects its "App Context" SystemMessage via the
   add_messages reducer, which appends it after the human turn. Anthropic
   requires system messages at the head; vLLM/SGLang (via LiteLLM) reject more
   than ONE system message even when consecutive ("System message must be at
   the beginning"). All system content is therefore merged into a single
   leading system message — valid for every backend.
2. Streaming a zero-length thinking block accumulates {type, signature} with
   no "thinking" key; Anthropic 400s on replay in the tool loop unless the
   empty text field is restored."""

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage, SystemMessage

_SYSTEM_TYPES = ("system", "developer")


def _repaired_message(message):
    if not isinstance(message, AIMessage) or not isinstance(message.content, list):
        return message
    if not any(
        isinstance(b, dict) and b.get("type") == "thinking" and "thinking" not in b
        for b in message.content
    ):
        return message
    content = [
        {**b, "thinking": ""}
        if isinstance(b, dict) and b.get("type") == "thinking" and "thinking" not in b
        else b
        for b in message.content
    ]
    return message.model_copy(update={"content": content})


def _system_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
    return ""


def _reordered(request: ModelRequest) -> ModelRequest:
    messages = [_repaired_message(m) for m in request.messages]
    system = [m for m in messages if getattr(m, "type", None) in _SYSTEM_TYPES]
    rest = [m for m in messages if getattr(m, "type", None) not in _SYSTEM_TYPES]
    if not system:
        if messages == list(request.messages):
            return request
        return request.override(messages=messages)
    extras = "\n\n".join(t for t in (_system_text(m) for m in system) if t)
    if request.system_message is not None:
        base = _system_text(request.system_message)
        merged_content = f"{base}\n\n{extras}" if extras else base
        return request.override(
            messages=rest, system_message=SystemMessage(content=merged_content)
        )
    return request.override(messages=[SystemMessage(content=extras), *rest])


class ConsecutiveSystemMessagesMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        return handler(_reordered(request))

    async def awrap_model_call(self, request, handler):
        return await handler(_reordered(request))
