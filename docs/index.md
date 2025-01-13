# Smelter Optimisation

A Python implementation of a smelter optimization problem that uses heuristic methods to optimise pot arrangements in crucibles for maximum efficiency.

## Features
- Next Ascent Solver implementation for optimisation
- Customizable neighborhood rules for solution space exploration
- Visualization of optimisation progress
- Configurable solver parameters

## Installation

From github
```bash
git clone https://github.com/gideon-m/smelter-optimisation.git
cd smelter-optimisation

pipx install poetry
poetry install

# or for development 
poetry install -e . # to install in editable mode
```

## Usage

Basic usage example:

```python
from smelter_optimisation.neighbourhood_rule import SwapTwoPotsRule
from smelter_optimisation.solver import NextAscentSolver
from smelter_optimisation.utils import load_initial_solution

# Load initial solution from CSV
initial_solution = load_initial_solution()

# Initialise solver with desired parameters
solver = NextAscentSolver(
    neighbourhood=SwapTwoPotsRule(),
    verbose=True,
    max_iter=500
)

# Run optimisation
solver.run_solver(initial_solution)

# Get optimised solution
optimal_solution, optimal_value = solver.solution()

# Plot objective function vs number of iterations
solver.plot_objective()
```

## Requirements
- Python 3.10+
- Poetry
- pandas
- matplotlib
- numpy

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
