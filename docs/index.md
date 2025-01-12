# Smelter Optimisation

A Python implementation of a smelter optimization problem that uses heuristic methods to optimise pot arrangements in crucibles for maximum efficiency.

## Features
- Next Ascent Solver implementation for optimization
- Customizable neighborhood rules for solution space exploration
- Visualization of optimization progress
- Configurable solver parameters

## Installation

```bash
pipx install poetry
poetry install
```

## Usage

Basic usage example:

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

## Testing 
```bash
poetry run pytest
```

## Requirements
- Python 3.10+
- Poetry
- pandas
- matplotlib
- numpy

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
