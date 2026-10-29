#!/usr/bin/env bash
#
# Development orchestrator for the Safe-Room Terminal monorepo.
#
#   ./scripts/dev.sh [--no-seed] [--no-db] [--no-frontend] [--no-backend]
#
# Brings up PostgreSQL (Docker), applies the demo data when the database is
# empty, then runs the FastAPI backend and the Vite dev server side by side with
# prefixed logs. Ctrl+C stops both processes; PostgreSQL keeps running.
#
# Ports: BACKEND_PORT (8000), FRONTEND_PORT (5173).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
COMPOSE_SERVICE="postgres"
DB_WAIT_TIMEOUT_SECONDS=60

RUN_SEED=1
RUN_DB=1
RUN_BACKEND=1
RUN_FRONTEND=1

SERVICE_PIDS=()
STOPPED=0

# --- output ----------------------------------------------------------------
COLOR_RESET='\033[0m'

log() {
  printf '\033[1;36m[terminal]\033[0m %s\n' "$*"
}

warn() {
  printf '\033[1;33m[terminal]\033[0m %s\n' "$*" >&2
}

fail() {
  printf '\033[1;31m[terminal]\033[0m %s\n' "$*" >&2
  exit 1
}

usage() {
  # Prints the descriptive comment block that follows the shebang, so the help
  # text lives next to the code it documents and never goes stale.
  awk 'NR > 2 && /^#/ { sub(/^# ?/, ""); print; next } NR > 2 { exit }' "${BASH_SOURCE[0]}"
}

parse_arguments() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --no-seed) RUN_SEED=0 ;;
      --no-db) RUN_DB=0 ;;
      --no-backend) RUN_BACKEND=0 ;;
      --no-frontend) RUN_FRONTEND=0 ;;
      -h | --help)
        usage
        exit 0
        ;;
      *) fail "Неизвестный аргумент: $1 (см. --help)" ;;
    esac
    shift
  done
}

# --- preflight -------------------------------------------------------------
PYTHON="$ROOT_DIR/.venv/bin/python"

require_python_environment() {
  [ -x "$PYTHON" ] || fail "Нет .venv. Сначала выполните: make install"
}

require_frontend_dependencies() {
  if [ -d "$ROOT_DIR/frontend/node_modules" ]; then
    return
  fi
  log "Зависимости фронтенда не установлены — ставлю (npm install)…"
  npm --prefix "$ROOT_DIR/frontend" install
}

ensure_env_file() {
  [ -f "$ROOT_DIR/.env" ] && return
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  warn "Создан .env из .env.example — впишите AI_API_KEY, иначе бэкенд не стартует."
}

warn_if_provider_key_is_missing() {
  local key
  key="$(grep -hsE '^(AI_API_KEY|OPENAI_API_KEY|ROUTERAI_API_KEY)=' "$ROOT_DIR/.env" 2>/dev/null |
    head -n 1 | cut -d= -f2- || true)"

  if [ -z "${AI_API_KEY:-${OPENAI_API_KEY:-${ROUTERAI_API_KEY:-}}}" ] &&
    { [ -z "$key" ] || [ "$key" = "sk-your-key-here" ]; }; then
    warn "Ключ провайдера не найден: укажите AI_API_KEY в .env — без него бэкенд не запустится."
  fi
}

port_is_busy() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

require_free_port() {
  port_is_busy "$1" &&
    fail "Порт $1 уже занят. Остановите процесс на нём, задайте другой порт (BACKEND_PORT/FRONTEND_PORT) или запустите без этого сервиса (--no-backend / --no-frontend)."
  return 0
}

# --- services --------------------------------------------------------------
prefix_stream() {
  awk -v label="$1" -v color="$2" \
    '{ printf "\033[%sm%-9s\033[0m | %s\n", color, label, $0; fflush() }'
}

start_service() {
  local label="$1"
  local color="$2"
  shift 2

  "$@" > >(prefix_stream "$label" "$color") 2>&1 &
  SERVICE_PIDS+=("$!")
}

kill_tree() {
  local pid="$1"
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}

stop_services() {
  local pid
  trap - INT TERM

  # Ctrl+C can arrive while the watchdog is still polling for a dead process,
  # so the teardown has to be safe to run twice.
  if [ "$STOPPED" -eq 1 ]; then
    return 0
  fi
  STOPPED=1

  log "Останавливаю сервисы приложения…"
  for pid in "${SERVICE_PIDS[@]:-}"; do
    kill_tree "$pid"
  done
  wait 2>/dev/null || true
}

finish() {
  stop_services
  log "PostgreSQL оставлен запущенным. Остановить: make db-down"
}

# --- steps -----------------------------------------------------------------
start_database() {
  log "Поднимаю PostgreSQL + pgvector…"
  docker compose up -d "$COMPOSE_SERVICE" >/dev/null
  wait_for_database
}

wait_for_database() {
  local container
  local waited=0

  while [ "$waited" -lt "$DB_WAIT_TIMEOUT_SECONDS" ]; do
    container="$(docker compose ps -q "$COMPOSE_SERVICE" 2>/dev/null || true)"
    if [ -n "$container" ] &&
      [ "$(docker inspect -f '{{.State.Health.Status}}' "$container" 2>/dev/null)" = "healthy" ]; then
      log "PostgreSQL готов (порт 5432)."
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done

  fail "PostgreSQL не стал healthy за ${DB_WAIT_TIMEOUT_SECONDS}s. Проверьте: docker compose logs $COMPOSE_SERVICE"
}

seed_database() {
  log "Проверяю демо-данные (seed.py идемпотентен)…"
  "$PYTHON" seed.py || fail "Не удалось наполнить базу. Проверьте ключ провайдера в .env."
}

start_backend() {
  require_free_port "$BACKEND_PORT"
  log "Бэкенд: http://localhost:$BACKEND_PORT (docs: /docs)"
  start_service "backend" "33" \
    "$ROOT_DIR/.venv/bin/uvicorn" app.main:app --reload --port "$BACKEND_PORT"
}

start_frontend() {
  require_free_port "$FRONTEND_PORT"
  log "Фронтенд: http://localhost:$FRONTEND_PORT"
  start_service "frontend" "32" \
    npm --prefix "$ROOT_DIR/frontend" run dev -- --port "$FRONTEND_PORT"
}

watch_services() {
  local pid
  while :; do
    if [ "$STOPPED" -eq 1 ]; then
      # An interactive Ctrl+C already tore everything down.
      return 0
    fi

    for pid in "${SERVICE_PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        warn "Процесс $pid завершился — останавливаю остальные."
        finish
        return 1
      fi
    done
    sleep 1
  done
}

# --- entry point -----------------------------------------------------------
main() {
  parse_arguments "$@"
  require_python_environment

  if [ "$RUN_FRONTEND" -eq 1 ]; then
    require_frontend_dependencies
  fi

  ensure_env_file
  warn_if_provider_key_is_missing

  trap stop_services INT TERM

  if [ "$RUN_DB" -eq 1 ]; then
    start_database
  fi
  if [ "$RUN_SEED" -eq 1 ]; then
    seed_database
  fi
  if [ "$RUN_BACKEND" -eq 1 ]; then
    start_backend
  fi
  if [ "$RUN_FRONTEND" -eq 1 ]; then
    start_frontend
  fi

  if [ "${#SERVICE_PIDS[@]}" -eq 0 ]; then
    log "Нечего запускать: все сервисы отключены флагами."
    return 0
  fi

  log "Всё поднято. Ctrl+C — остановить бэкенд и фронтенд."
  watch_services || return 1

  finish
}

main "$@"
