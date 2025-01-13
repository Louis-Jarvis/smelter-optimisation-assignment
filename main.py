"""
Demonstration of running the smelter optimisation problem.
"""

from smelter_optimisation.neighbourhood_rule import SwapTwoPotsRule
from smelter_optimisation.solver import NextAscentSolver
from smelter_optimisation.utils import load_initial_solution


def main() -> None:
    xi = load_initial_solution()

    solver = NextAscentSolver(neighbourhood=SwapTwoPotsRule(), verbose=True)

    solver.optimise(xi, max_iter=500)
    x_optim, f_optim = solver.get_solution()

    print(f"x_optim: {x_optim}")
    print(f"f_optim: {f_optim}")

    solver.plot_objective()

if __name__ == "__main__":
    main()
