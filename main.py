import pathlib
import pandas as pd

from smelter_optimisation.neighbourhood_rule import SwapTwoPotsRule
from smelter_optimisation.solver import NextAscentSolver
from smelter_optimisation.utils import create_init_sol


def main() -> None:
    xi = create_init_sol(pd.read_csv(pathlib.Path("data/initial_solution.csv")))

    solver = NextAscentSolver(neighbourhood=SwapTwoPotsRule(), verbose=True)

    x_optim, f_optim = solver.run_solver(xi, max_iter=500)

    solver.plot_objective()

    return


if __name__ == "__main__":
    main()
