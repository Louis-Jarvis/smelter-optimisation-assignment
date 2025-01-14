# Smelter Optimisation

[![Python package](https://github.com/louis-jarvis/smelter-optimisation-assignment/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/louis-jarvis/smelter-optimisation-assignment/actions/workflows/ci-cd.yml)
[![Documentation Status](https://img.shields.io/badge/docs-mkdocs-blue)](https://your-username.github.io/smelter-optimisation/)
[![Version](https://img.shields.io/badge/version-0.1.0-brightgreen)](https://github.com/louis-jarvis/smelter-optimisation-assignment)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/louis-jarvis/smelter-optimisation-assignment/blob/main/LICENSE)

A Python implementation of a smelter optimization problem that uses heuristic methods to optimize pot arrangements in crucibles for maximum efficiency.

Documentation: https://louis-jarvis.github.io/smelter-optimisation-assignment/


## Features
- **Next Ascent Solver** implementation for optimization
- **Customizable neighborhood rules** for solution space exploration
- **Visualization of optimization progress**
- **Configurable solver parameters**

## Installation

```bash
git clone https://github.com/gideon-m/smelter-optimisation.git
cd smelter-optimisation

pipx install poetry
poetry install
```

## Usage

For example to run the next ascent solver using a swap two pots neighbourhood rule:

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

# Run optimisation and get optimised solution
solver.optimise(initial_solution, max_iter=500)
optimal_solution, optimal_value = solver.get_solution()

print(f"Optimal value: ${optimal_value:.2f}")
print(f"Optimal solution:")
print_solution(optimal_solution)

# Plot objective function vs number of iterations
solver.plot_objective()
```

## Testing 
Tests are run using [pytest](https://docs.pytest.org/en/latest/) and are located in the `tests` directory.
```bash
poetry run pytest

# or with makefile
make test
```

## Documentation
Documentation is built using [mkdocs](https://www.mkdocs.org/) and is hosted on GitHub Pages.
```bash
# this will generate the docs and serve them locally
make docs 
```

## Requirements
- Python 3.10+
- `Poetry`
- `pandas`
- `matplotlib`
- `numpy`

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

