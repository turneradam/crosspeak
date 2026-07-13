default:
    @just --list

install:
    uv sync

test:
    uv run pytest -q

lint:
    uv run ruff check src tests

format:
    uv run ruff format src tests

check:
    uv run ruff check src tests
    uv run ruff format --check src tests
    uv run pytest -q

build:
    uv build

release version:
    git tag v{{version}}
    git push origin v{{version}}
    gh release create v{{version}} --generate-notes

docs-serve:
    uv run mkdocs serve

docs-build:
    uv run mkdocs build --strict