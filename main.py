from smelter_optimisation.solver.solver import NextAscentSolver
from smelter_optimisation.solver.neighbourhood_rule import Swap2PotsRule
from smelter_optimisation.utils import create_init_sol
from smelter_optimisation.plot_objective import plot_objective


def main() -> None:
    xi, fi = create_init_sol()

    solver = NextAscentSolver(
        neighbourhood=Swap2PotsRule(verbose=True), verbose=True, max_iter=100
    )

    solver.run_solver(xi, fi)

    x_optim, f_optim = solver.solution

    plot_objective(solution=solver)

    return


if __name__ == "__main__":
    pass
