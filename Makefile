# Define the path for the poetry binary
POETRY_BIN=$(shell which poetry)

# Define the Python environment
PYTHON_ENV=$(shell which python)

# Check if Poetry is installed
POETRY_INSTALLED=$(if $(POETRY_BIN),yes,no)

# Default target
.PHONY: install install-poetry install-deps test lint

# Install Poetry if it's not already installed
install-poetry:
	@if [ "$(POETRY_INSTALLED)" = "no" ]; then \
		echo "Poetry not found. Installing..."; \
		curl -sSL https://install.python-poetry.org | python3 -; \
		else \
		echo "Poetry is already installed."; \
	fi

# Install dependencies using Poetry
install-deps: install-poetry
	poetry install

install-ruff: install-poetry
	poetry add ruff

# Run tests with Poetry (using pytest as an example, can be changed)
test: install-deps
	poetry run pytest

lint: install-ruff
	ruff check 

# Clean all poetry environments
clean:
	rm -rf .venv
	poetry env remove $(python) || true

# Help command to list available make commands
help:
	@echo "Available commands:"
	@echo "  install-deps    Install dependencies using poetry"
	@echo "  test       Run tests using poetry and pytest"
	@echo "  install-poetry  Install Poetry if it's not installed"
	@echo "  clean           Clean all poetry virtual environments"
