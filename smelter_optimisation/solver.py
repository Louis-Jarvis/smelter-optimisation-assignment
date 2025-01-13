"""This module contains different heuristics used to solvers for the smelter optimisation problem.

This includes:

- **Next Ascent Solver**:
A greedy algorithm that chooses the first neighbour that results in an improvement in the objective function.
"""

import logging
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from smelter_optimisation.config import TOL
from smelter_optimisation.models import Crucible
from smelter_optimisation.neighbourhood_rule import NeighbourhoodRule
from smelter_optimisation.utils import calc_crucible_value

logger = logging.getLogger(__name__)


def _calc_delta_fx(x0, x_new, c1, c2) -> float:
    # since the solution is additive we can calculate the change in
    delta_1 = calc_crucible_value(x_new[c1]) - calc_crucible_value(x0[c1])
    delta_2 = calc_crucible_value(x_new[c2]) - calc_crucible_value(x0[c2])
    return delta_1 + delta_2


class SmeltingOptimisationSolver(ABC):
    """Smelter optimisation problem solver heuristic."""

    @abstractmethod
    def run_solver(self, initial_solution: list[Crucible]):
        pass

    @abstractmethod
    def plot_objective(self):
        pass

    @abstractmethod
    def _calculate_objective_value(self, x):
        pass


class NextAscentSolver(SmeltingOptimisationSolver):
    """Solver utilising a Next Ascent Greedy Heuristic.
    
    Args:
        neighbourhood (NeighbourhoodRule): neighborhood rule that defines neighbours
            adjacent to the current pot.
        verbose (bool, optional): verbosity. Defaults to False.
        max_iter (int, optional): maximum iteration count. Defaults to 5000.
    
    Examples:
        >>> solver = NextAscentSolver(neighbourhood=SwapTwoPotsRule(), verbose=True, max_iter=500)
    """

    def __init__(
        self,
        neighbourhood: NeighbourhoodRule,
        verbose: bool = False,
        max_iter: int = 5000,
    ) -> None:
        self.neighbourhood_rule = neighbourhood
        self.verbose = verbose
        self.max_iter = max_iter
        self.converged = False

        self.objective_value_history = []

        self._num_iter = 0
        self._current_solution = None
        self._current_value = None

    #TODO pass in max iter as an argument here
    def run_solver(self, initial_solution: list[Crucible]) -> tuple[list[Crucible], float]:
        """Determine the optimal solution based on the next ascent solver.

        Args:
            initial_solution (list[Crucible]): an initial array of crucibles.

        Returns:
            tuple[List[Pot], float]: A tuple containing:
                - The current solution (List[Pot])
                - The objective value (float)

        Examples:
            >>> import pathlib
            >>> import pandas as pd
            >>>
            >>> from smelter_optimisation.neighbourhood_rule import SwapTwoPotsRule
            >>> from smelter_optimisation.solver import NextAscentSolver
            >>> from smelter_optimisation.utils import create_init_sol
            >>>
            >>> xi = create_init_sol(pd.read_csv(pathlib.Path("data/initial_solution.csv")))
            >>> solver = NextAscentSolver(neighbourhood=SwapTwoPotsRule(), verbose=True, max_iter=500)
            >>> x_optim, f_optim = solver.run_solver(xi)
        """
        optimal_value = self._calculate_objective_value(initial_solution)

        self._current_solution = initial_solution
        self._current_value = optimal_value

        self.objective_value_history.append(self._current_value)

        while True:
            logger.info("New neighbourhood")
            neighbourhood = self.neighbourhood_rule.generate_neighbours(self._current_solution)

            # explore neighbourhood for increase in objective value
            for neighbour in neighbourhood:
                self._num_iter += 1

                # evaluate neighbour
                new_objective_value = self._calculate_objective_value(neighbour)
                self.objective_value_history.append(new_objective_value)

                # check for improvement
                if (new_objective_value - self._current_value) > TOL:
                    if self.verbose:
                        logger.info(f"Accept Swap: current best fx: {new_objective_value:.4f}")

                    self._current_solution = neighbour
                    self._current_value = new_objective_value
                    break

                if self._num_iter == self.max_iter:
                    warnings.warn(f"Max iterations ({self.max_iter}) reached.", stacklevel=1)
                    return self._current_solution, self._current_value

                # best value in neighbourhood
                optimal_value = self._current_value

            # check convergence
            if np.abs(self._current_value - optimal_value) < TOL:
                self.converged = True
                logger.info("Converged")
                return self._current_solution, self._current_value

    #TODO refactor this
    def _calculate_objective_value(self, x):
        return np.sum([calc_crucible_value(crucible) for crucible in x])

    def plot_objective(self) -> None:
        """Plot the objective function against the number of function evaluations."""
        _, ax = plt.subplots()

        x = np.arange(0, len(self.objective_value_history))
        ax.plot(x, self.objective_value_history)

        ax.set(xlabel="Function Evaluations", ylabel="Objective Function Value f(x)")

        plt.show()
