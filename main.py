"""
Demonstration of running the smelter optimisation problem.
"""

import pathlib
import pandas as pd

from smelter_optimisation.neighbourhood_rule import SwapTwoPotsRule
from smelter_optimisation.solver import NextAscentSolver
from smelter_optimisation.utils import create_init_sol


def main() -> None:
    xi = create_init_sol(pd.read_csv(pathlib.Path("data/initial_solution.csv")))

    solver = NextAscentSolver(neighbourhood=SwapTwoPotsRule(), verbose=True)

    solver.optimise(xi, max_iter=500)
    x_optim, f_optim = solver.get_solution()

    print(f"x_optim: {x_optim}")
    print(f"f_optim: {f_optim}")

    solver.plot_objective()

if __name__ == "__main__":
    main()
