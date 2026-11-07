# Safe-Room Terminal — единая точка входа для разработки.
#
# make dev          # Qdrant + seed + бэкенд + фронтенд одной командой
# make help         # список всех целей

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
FRONTEND := frontend

.PHONY: help install install-python install-frontend vectors vectors-down seed seed-reset \
	reingest import-lore-csv corpus-refresh corpus-translate assets user dev dev-backend \
	dev-frontend test lint typecheck eval check build clean

help: ## Показать доступные команды
	@printf '\nSafe-Room Terminal\n\n'
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@printf '\n'

install: install-python install-frontend ## Установить всё: .venv, зависимости Python и Node

install-python: ## Создать .venv и поставить зависимости Python
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt

install-frontend: ## Установить зависимости фронтенда
	npm --prefix $(FRONTEND) install

vectors: ## Поднять Qdrant в Docker (векторы и образы живут здесь)
	docker compose up -d qdrant

vectors-down: ## Остановить векторный узел (данные сохраняются в томе)
	docker compose down

seed: ## Наполнить хранилище: образы + корпус из снимка (идемпотентно)
	$(PYTHON) seed.py

seed-reset: ## Пересоздать коллекции и наполнить хранилище заново (учётные записи не трогает)
	$(PYTHON) seed.py --reset

user: ## Завести учётную запись: make user EMAIL=operator@l4d.dev [ROLE=admin] [NAME=Оператор]
	@test -n "$(EMAIL)" || { echo "Укажите адрес: make user EMAIL=operator@l4d.dev"; exit 1; }
	$(PYTHON) -m scripts.create_user --email "$(EMAIL)" --role "$(or $(ROLE),admin)" $(if $(NAME),--name "$(NAME)")

reingest: ## Переиндексировать архив: очистить lore_chunks и залить снимок заново
	$(PYTHON) seed.py --reingest

import-lore-csv: ## Перенести архив из выгрузки pgvector в Qdrant (без повторных эмбеддингов)
	$(PYTHON) -m scripts.import_lore_csv

corpus-refresh: ## Перечитать вики и пересобрать снимок корпуса (без хранилища)
	$(PYTHON) seed.py --refresh --corpus-only

corpus-translate: ## То же плюс нормализация языка корпуса (долго, платно; нужно только для нерусского источника)
	$(PYTHON) seed.py --refresh --corpus-only --translate

assets: ## Пересобрать ассеты интерфейса: портреты выживших, фон и знак из вики
	$(PYTHON) scripts/fetch_brand_assets.py

dev: ## Запустить весь проект: Qdrant, seed, бэкенд (:8000) и фронтенд (:5173)
	./scripts/dev.sh

dev-backend: ## Запустить только бэкенд (Qdrant должен быть поднят и наполнен)
	.venv/bin/uvicorn app.main:app --reload

dev-frontend: ## Запустить только фронтенд
	npm --prefix $(FRONTEND) run dev

test: ## Прогнать тесты бэкенда
	$(PYTHON) -m pytest

eval: ## Оценить качество ответов LLM-судьёй (нужны поднятый Qdrant, корпус и AI_API_KEY)
	$(PYTHON) -m scripts.evaluate

lint: ## Проверить код линтером ruff
	$(PYTHON) -m ruff check .

typecheck: ## Проверить типы через mypy
	$(PYTHON) -m mypy

check: lint typecheck test ## Линтер + типы + тесты

build: ## Собрать фронтенд в frontend/dist
	npm --prefix $(FRONTEND) run build

clean: ## Удалить кэши, сборку и node_modules
	rm -rf .pytest_cache .mypy_cache .ruff_cache $(FRONTEND)/dist $(FRONTEND)/node_modules
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
