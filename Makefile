PYTHON := python3
PIP := pip

.PHONY: up down logs ps test lint dev ingest ask bootstrap

# Use the root docker-compose file
COMPOSE := docker compose -f docker-compose.yml

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200

ps:
	$(COMPOSE) ps

test:
	pytest -q

lint:
	ruff check . || true

dev:
	uvicorn api.main:app --port 8080 --reload

# Make helpers for quick manual tests
ingest:
	@if [ -z "$(PDF)" ]; then echo "Usage: make ingest PDF=<url>"; exit 1; fi; \
	curl -sS -X POST http://localhost:8080/ingest -H "Content-Type: application/json" -d '{"pdf_url":"'"$(PDF)"'"}' | sed -E 's/.{0}//'

ask:
	@if [ -z "$(Q)" ]; then echo "Usage: make ask Q=\"Your question\""; exit 1; fi; \
	curl -N -X POST http://localhost:8080/ask -H "Content-Type: application/json" -d '{"question":"'"$(Q)"'"}'

bootstrap:
	bash scripts/dev_bootstrap.sh
