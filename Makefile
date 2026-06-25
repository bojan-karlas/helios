# HELIOS — main Makefile (development entrypoint).

MAKEFLAGS += --no-print-directory

SUMMARY := HELIOS computational-pathology pipeline. Scaffold with runnable seams and stubbed domain logic. Use uv for dependency management; src-layout means install-first.

MY_DIR_PATH := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT_DIR_PATH := $(realpath $(MY_DIR_PATH))

UV := uv

include $(ROOT_DIR_PATH)/dev/makefiles/show-help.mk

.PHONY: install
## Create the virtual environment and install helios (+ dev tools) via uv.
install:
	$(UV) sync --extra dev

.PHONY: test
## Run the test suite (incl. the end-to-end smoke test on synthetic fixtures).
test:
	$(UV) run pytest

.PHONY: smoke
## Run only the end-to-end pipeline smoke test.
smoke:
	$(UV) run pytest tests/test_smoke.py -v

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

.PHONY: fit
## Run the training pipeline (helios fit). Pass ARGS="--config ...".
fit:
	$(UV) run helios fit $(ARGS)

.PHONY: predict
## Run the inference pipeline (helios predict). Pass ARGS="--config ...".
predict:
	$(UV) run helios predict $(ARGS)

.PHONY: clean
## Remove caches and build artifacts (keeps .venv).
clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache **/__pycache__ build dist *.egg-info
