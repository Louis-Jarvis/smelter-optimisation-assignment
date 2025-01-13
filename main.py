from smelter_optimisation.neighbourhood_rule import SwapTwoPotsRule
from smelter_optimisation.solver import NextAscentSolver
from smelter_optimisation.utils import create_init_sol


def main() -> None:
    xi = create_init_sol()

    solver = NextAscentSolver(neighbourhood=SwapTwoPotsRule(), verbose=True, max_iter=500)

    x_optim, f_optim = solver.run_solver(xi)

    solver.plot_objective()

    return


if __name__ == "__main__":
    main()
