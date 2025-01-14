# Smelter Optimisation

A Python implementation of a smelter optimization problem that uses heuristic methods to optimise pot arrangements in crucibles for maximum price.

## Overview

The smelter optimisation problem involves arranging pots containing mixtures of aluminium and iron into crucibles to maximise their value. Each crucible contains three pots, and the value of a crucible depends on the average composition of aluminium and iron across its pots.

## Installation

From github:
```bash
git clone https://github.com/your-username/smelter-optimisation.git
cd smelter-optimisation

poetry install
```

## Quick Start

```python
from smelter_optimisation.neighbourhood_rule import SwapTwoPotsRule
from smelter_optimisation.solver import NextAscentSolver
from smelter_optimisation.utils import load_initial_solution

# Load initial solution from CSV
initial_solution = load_initial_solution()

# Initialise solver with desired parameters
solver = NextAscentSolver(
    neighbourhood=SwapTwoPotsRule(),
    verbose=True
)

# Run optimisation
solver.optimise(initial_solution, max_iter=500)
optimal_solution = solver.optimal_solution
optimal_value = solver.optimal_value

print(f"Optimal value: ${optimal_value:.2f}")
print(f"Optimal solution:")
print_solution(optimal_solution)

# Plot objective function vs number of iterations
solver.plot_objective()
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
