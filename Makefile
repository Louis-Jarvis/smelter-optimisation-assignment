.PHONY: clean test type-check docs help setup bootstrap

# Variables
POETRY = poetry run
PYTEST = $(POETRY) pytest
MYPY = $(POETRY) mypy
PACKAGE = smelter_optimisation

help:
	@echo "Available commands:"
	@echo "make setup     - Install all dependencies using poetry"
	@echo "make clean      - Remove Python file artifacts and cache directories"
	@echo "make test      - Run pytest"
	@echo "make type-check- Run mypy type checking"
	@echo "make docs      - Serve mkdocs documentation locally"
	@echo "make all       - Run setup, clean, type-check, test"

setup:
	@echo "Installing dependencies..."
	poetry install --with dev
	@echo "Installation complete!"

clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name "build" -exec rm -rf {} +
	@find . -type d -name "dist" -exec rm -rf {} +
	@echo "Clean complete!"

test: setup
	$(PYTEST) -v tests/

type-check: setup
	$(MYPY) $(PACKAGE)

docs: setup
	$(POETRY) mkdocs serve

doctest:
	$(POETRY) python -m doctest -v smelter_optimisation/*.py

all: clean type-check test docs