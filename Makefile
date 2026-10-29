# Safe-Room Terminal — единая точка входа для разработки.
#
# make dev          # Postgres + seed + бэкенд + фронтенд одной командой
# make help         # список всех целей

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
FRONTEND := frontend

.PHONY: help install install-python install-frontend db db-down seed seed-reset \
	reingest corpus-refresh corpus-translate assets dev dev-backend dev-frontend test lint typecheck \
	check build clean

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

db: ## Поднять PostgreSQL 16 + pgvector в Docker
	docker compose up -d postgres

db-down: ## Остановить контейнер базы (данные сохраняются в томе)
	docker compose down

seed: ## Наполнить базу: персонажи + корпус из снимка (идемпотентно)
	$(PYTHON) seed.py

seed-reset: ## Пересоздать схему и наполнить базу заново
	$(PYTHON) seed.py --reset

reingest: ## Переиндексировать архив: очистить lore_chunks и залить снимок заново
	$(PYTHON) seed.py --reingest

corpus-refresh: ## Перечитать вики и пересобрать снимок корпуса (без базы)
	$(PYTHON) seed.py --refresh --corpus-only

corpus-translate: ## То же плюс нормализация языка корпуса (долго, платно)
	$(PYTHON) seed.py --refresh --corpus-only --translate

assets: ## Пересобрать ассеты интерфейса: портреты выживших, фон и знак из вики
	$(PYTHON) scripts/fetch_brand_assets.py

dev: ## Запустить весь проект: база, seed, бэкенд (:8000) и фронтенд (:5173)
	./scripts/dev.sh

dev-backend: ## Запустить только бэкенд (база должна быть поднята)
	.venv/bin/uvicorn app.main:app --reload

dev-frontend: ## Запустить только фронтенд
	npm --prefix $(FRONTEND) run dev

test: ## Прогнать тесты бэкенда
	$(PYTHON) -m pytest

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
