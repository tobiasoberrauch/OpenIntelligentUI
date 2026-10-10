"""Message normalization for OpenAI-compatible open-model servers.

CopilotKitMiddleware appends an "App Context" SystemMessage to the
conversation. OpenAI accepts several system messages, but many open-model chat
templates served by vLLM, Ollama or llama.cpp (Mistral, Gemma, gpt-oss and
others) accept a single leading system message or raise a template error.
Folding every system message into the request's system prompt at model-call
time keeps graph state untouched and works with any template.
"""

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

_SYSTEM_TYPES = ("system", "developer")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        )
    return ""


def _merged(request: ModelRequest) -> ModelRequest:
    extra = [m for m in request.messages if getattr(m, "type", None) in _SYSTEM_TYPES]
    if not extra:
        return request
    rest = [m for m in request.messages if getattr(m, "type", None) not in _SYSTEM_TYPES]
    parts = [_text(request.system_message.content)] if request.system_message else []
    parts += [_text(m.content) for m in extra]
    content = "\n\n".join(part for part in parts if part)
    return request.override(messages=rest, system_message=SystemMessage(content=content))


class SingleSystemMessageMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        return handler(_merged(request))

    async def awrap_model_call(self, request, handler):
        return await handler(_merged(request))
