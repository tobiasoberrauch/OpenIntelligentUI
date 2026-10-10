"""Ephemeral provider credentials, scoped to the complete ASGI stream lifetime."""

from contextvars import ContextVar
from dataclasses import dataclass, field
import asyncio
import hashlib
import hmac
import json
import os

import httpx
from copilotkit import LangGraphAGUIAgent
from langsmith import tracing_context
from starlette.responses import JSONResponse


@dataclass(frozen=True)
class Credentials:
    openai: str = field(repr=False)
    # Optional: a visitor's Jev key selects Jev routing under ROUTER_BACKEND=auto.
    jev: str | None = field(default=None, repr=False)


current_credentials: ContextVar[Credentials | None] = ContextVar('provider_credentials', default=None)
SENSITIVE_HEADERS = {b'x-openai-api-key', b'x-jev-api-key'}


def parse_credentials(headers):
    values = {}
    for name, value in headers:
        name = name.lower()
        if name in SENSITIVE_HEADERS:
            if name in values:
                raise ValueError('invalid_keys')
            # Reject controls before trimming: HTTP whitespace is spaces only.
            if any(c < 32 or c > 126 for c in value):
                raise ValueError('invalid_keys')
            value = value.strip(b' ')
            if not value or len(value) > 4096 or any(c <= 32 for c in value):
                raise ValueError('invalid_keys')
            values[name] = value.decode('ascii')
    if not values:
        return None
    # The OpenAI key is required; a Jev key alone never falls back to server secrets.
    if b'x-openai-api-key' not in values:
        raise ValueError('invalid_keys')
    return Credentials(values[b'x-openai-api-key'], values.get(b'x-jev-api-key'))


def server_credentials_available():
    # Self-hosted OpenAI-compatible servers (vLLM, Ollama, llama.cpp …) often run
    # without keys; the base URL is the server-side configuration.
    if os.environ.get('LLM_BASE_URL', '').strip():
        return True
    model = os.environ.get('LLM_MODEL', 'chat-latest').strip()
    key = 'ANTHROPIC_API_KEY' if model.startswith('claude-') else 'OPENAI_API_KEY'
    return bool(os.environ.get(key, '').strip())


class CredentialsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        try:
            credentials = parse_credentials(scope.get('headers', []))
        except ValueError:
            return await JSONResponse({'ok': False, 'code': 'invalid_keys'}, status_code=400)(scope, receive, send)
        if scope['path'] == '/' and not credentials and not server_credentials_available():
            return await JSONResponse({'ok': False, 'code': 'credentials_required'}, status_code=401)(scope, receive, send)
        clean_scope = {**scope, 'headers': [(k, v) for k, v in scope.get('headers', []) if k.lower() not in SENSITIVE_HEADERS]}
        token = current_credentials.set(credentials)
        try:
            # BYOK provider calls and graph runs must never enter hosted traces.
            with tracing_context(enabled=False if credentials else None):
                await self.app(clean_scope, receive, send)
        finally:
            current_credentials.reset(token)


def private_thread_id(thread_id):
    credentials = current_credentials.get()
    if credentials:
        key = hashlib.sha256(json.dumps([credentials.openai, credentials.jev]).encode()).digest()
        namespace = 'byok'
    else:
        key = b'open-intelligent-ui-server-threads-v1'
        namespace = 'server'
    return namespace + ':' + hmac.new(key, thread_id.encode(), hashlib.sha256).hexdigest()


class CredentialScopedAgent(LangGraphAGUIAgent):
    async def run(self, input):
        original = input.thread_id
        scoped = input.model_copy(update={'thread_id': private_thread_id(original)})
        async for event in super().run(scoped):
            # The wire protocol retains the client ID; only graph storage sees the private ID.
            if hasattr(event, 'thread_id'):
                event = event.model_copy(update={'thread_id': original})
            yield event


async def validate_credentials(credentials):
    """Exercise the visitor's keys; never forward response bodies or exceptions.

    The OpenAI key serves answers and, without a Jev key or a self-hosted router
    endpoint, visualization routing. An optional Jev key is checked against Jev.
    """
    from src.visualization_router import JEV_URL, jev_request_options, router_backend

    backend = router_backend()
    if backend == 'jev' and not credentials.jev:
        # This deployment always routes with Jev; visitors never spend the server's key.
        return JSONResponse({'ok': False, 'code': 'jev_required'}, status_code=400)

    async def check(provider, url, options):
        try:
            async with httpx.AsyncClient(timeout=12, follow_redirects=False, trust_env=False) as client:
                response = await client.post(url, **options)
            if response.is_success:
                return None
            if response.status_code in {400, 401, 403, 404}:
                return provider + '_invalid'
        except httpx.HTTPError:
            pass
        return 'provider_unavailable'

    checks = [check('openai', 'https://api.openai.com/v1/chat/completions', {
        'headers': {'Authorization': f'Bearer {credentials.openai}'},
        'json': {'model': 'chat-latest', 'messages': [{'role': 'user', 'content': 'Reply OK.'}], 'max_completion_tokens': 8},
    })]
    # Only check a Jev key that this deployment can actually use.
    if credentials.jev and backend in {'auto', 'jev'}:
        token = current_credentials.set(credentials)
        try:
            options = jev_request_options([{'role': 'human', 'content': 'Hello'}])
        finally:
            current_credentials.reset(token)
        checks.append(check('jev', JEV_URL, options))
    results = await asyncio.gather(*checks)
    error = next((result for result in results if result), None)
    if error:
        return JSONResponse({'ok': False, 'code': error}, status_code=503 if error == 'provider_unavailable' else 401)
    return JSONResponse({'ok': True})
