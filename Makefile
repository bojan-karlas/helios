# HELIOS — main Makefile (development entrypoint).

MAKEFLAGS += --no-print-directory

SUMMARY := HELIOS computational-pathology pipeline. Pure components composed by CLI-callable stages (prep/fit/predict/report). Use uv for dependency management; src-layout means install-first.

MY_DIR_PATH := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT_DIR_PATH := $(realpath $(MY_DIR_PATH))

UV := uv
UV_RUN := $(UV) run $(if $(wildcard $(ROOT_DIR_PATH)/.env),--env-file "$(ROOT_DIR_PATH)/.env") --

include $(ROOT_DIR_PATH)/dev/makefiles/show-help.mk

.PHONY: install
## Create the virtual environment and install helios (+ dev tools) via uv.
install:
	$(UV) sync --extra dev

.PHONY: shell
## Open a Bash sub-shell with .env, the virtualenv, and a (helios) prompt.
shell:
	@$(UV_RUN) bash --rcfile "$(ROOT_DIR_PATH)/dev/shell.bash" -i

.PHONY: test
## Run the test suite.
test:
	@$(UV_RUN) pytest

.PHONY: lint
## Run ruff (lint) and mypy (type-check).
lint:
	@$(UV_RUN) ruff check src tests
	@$(UV_RUN) mypy

.PHONY: format
## Auto-format and auto-fix with ruff.
format:
	@$(UV_RUN) ruff format src tests
	@$(UV_RUN) ruff check --fix src tests

.PHONY: configs
## Regenerate configs/default.yaml from stage signatures.
configs:
	@$(UV_RUN) python scripts/gen_configs.py

.PHONY: clean
## Remove caches and build artifacts (keeps .venv).
clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache **/__pycache__ build dist *.egg-info
