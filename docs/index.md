# Smelter Optimisation

A Python implementation of a smelter optimization problem that uses heuristic methods to optimise pot arrangements in crucibles for maximum efficiency.

## Overview

The smelter optimisation problem involves arranging pots containing mixtures of aluminium and iron into crucibles to maximize their value. Each crucible contains three pots, and the value of a crucible depends on the average composition of aluminium and iron across its pots.

## Quick Start

```python
import pathlib
import pandas as pd
from smelter_optimisation.neighbourhood_rule import SwapTwoPotsRule
from smelter_optimisation.solver import NextAscentSolver
from smelter_optimisation.utils import create_init_sol

# Load initial solution from CSV
initial_solution = create_init_sol(
    pd.read_csv(pathlib.Path("data/initial_solution.csv"))
)

# Initialise solver with desired parameters
solver = NextAscentSolver(
    neighbourhood=SwapTwoPotsRule(),
    verbose=True
)

# Run optimisation
solver.optimise(initial_solution, max_iter=500)
optimal_solution, optimal_value = solver.get_solution()

# Plot objective function vs number of iterations
solver.plot_objective()
```

## Installation

```bash
# Install using poetry
git clone https://github.com/your-username/smelter-optimisation.git
cd smelter-optimisation
poetry install
```

## Development

```bash
# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Run type checking
poetry run mypy smelter_optimisation
```

## Requirements
- Python 3.10+
- Poetry
- pandas
- matplotlib
- numpy

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
