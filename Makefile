.PHONY: setup data test lint run-small run-full

setup:
	uv sync --extra dev

data:
	uv run forecast-scarce data all

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

run-small:
	@echo "not implemented until the runner stage"

run-full:
	@echo "not implemented until the runner stage"
