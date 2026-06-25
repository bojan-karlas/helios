# HELIOS — main Makefile (development entrypoint).

MAKEFLAGS += --no-print-directory

SUMMARY := HELIOS computational-pathology pipeline. Pure components composed by CLI-callable stages (prep/fit/predict/report). Use uv for dependency management; src-layout means install-first.

MY_DIR_PATH := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT_DIR_PATH := $(realpath $(MY_DIR_PATH))

UV := uv

include $(ROOT_DIR_PATH)/dev/makefiles/show-help.mk

.PHONY: install
## Create the virtual environment and install helios (+ dev tools) via uv.
install:
	$(UV) sync --extra dev

.PHONY: shell
## Open a sub-shell with the project virtualenv activated.
shell:
	$(UV) run $${SHELL:-bash}

.PHONY: test
## Run the test suite.
test:
	$(UV) run pytest

.PHONY: lint
## Run ruff (lint) and mypy (type-check).
lint:
	$(UV) run ruff check src tests
	$(UV) run mypy

.PHONY: format
## Auto-format and auto-fix with ruff.
format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

.PHONY: configs
## Regenerate configs/default.yaml from stage signatures.
configs:
	$(UV) run python scripts/gen_configs.py

.PHONY: clean
## Remove caches and build artifacts (keeps .venv).
clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache **/__pycache__ build dist *.egg-info
