import pathlib
import pandas as pd

from smelter_optimisation.neighbourhood_rule import Swap2PotsRule
from smelter_optimisation.solver import NextAscentSolver
from smelter_optimisation.utils import create_init_sol


def main() -> None:
    xi = create_init_sol(pd.read_csv(pathlib.Path("data/initial_solution.csv")))

    solver = NextAscentSolver(neighbourhood=Swap2PotsRule(), verbose=True, max_iter=500)

    solver.run_solver(xi)

    x_optim, f_optim = solver.solution

    solver.plot_objective()

    return


if __name__ == "__main__":
    main()
