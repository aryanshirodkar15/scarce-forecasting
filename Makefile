.PHONY: setup data test lint run-small run-full figures-small figures paper

setup:
	uv sync --extra dev

data:
	uv run forecast-scarce data all

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

run-small:
	uv run forecast-scarce run --config configs/small.yaml

run-full:
	uv run forecast-scarce run --config configs/full.yaml

paper:
	cd paper && tectonic -X compile main.tex

figures-small:
	uv run forecast-scarce figures --results results/small

figures:
	uv run forecast-scarce figures --results results/full --out paper/figures
