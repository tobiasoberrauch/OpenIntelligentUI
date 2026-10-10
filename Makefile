.PHONY: help install setup dev dev-app dev-agent dev-mcp build lint clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (Node + Python)
	pnpm install

setup: install ## Full setup: install deps and create .env from template
	@if [ ! -f apps/agent/.env ]; then \
		printf '%s\n' '# Open models: any OpenAI-compatible server (vLLM, Ollama, llama.cpp, LiteLLM).' '# Ollama example; create qwen3-coder-32k as described in docs/open-models.md.' 'LLM_BASE_URL=http://localhost:11434/v1' 'LLM_MODEL=qwen3-coder-32k' '# Optional: a smaller, faster model for visualization routing.' '# ROUTER_MODEL=qwen2.5:3b' '# Hosted OpenAI instead: remove LLM_BASE_URL, set LLM_MODEL=chat-latest and OPENAI_API_KEY.' 'OPENAI_API_KEY=' '# Optional: TypeSafe Jev for visualization routing (used automatically when set).' 'TYPESAFE_API_KEY=' > apps/agent/.env; \
		echo "Created apps/agent/.env — set LLM_BASE_URL/LLM_MODEL for your model server (see docs/open-models.md)"; \
	else \
		echo "apps/agent/.env already exists, skipping"; \
	fi

dev: ## Start all services (frontend + agent + mcp)
	pnpm dev

dev-app: ## Start Next.js frontend only (http://localhost:3000)
	pnpm dev:app

dev-agent: ## Start LangGraph agent only (http://localhost:8123)
	pnpm dev:agent

dev-mcp: ## Start MCP server only
	pnpm dev:mcp

build: ## Build all apps
	pnpm build

lint: ## Lint all apps
	pnpm lint

clean: ## Clean build artifacts
	pnpm clean
