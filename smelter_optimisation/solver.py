"""This module contains different heuristics used to solvers for the smelter optimisation problem.

This includes:

- **Next Ascent Solver**:
A greedy algorithm that chooses the first neighbour that results in an improvement in the objective function.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from smelter_optimisation import config
from smelter_optimisation.models import Crucible, Pot
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
    
    :param neighbourhood: neighborhood rule that defines neighbours
    adjacent to the current pot.
    :type neighbourhood: NeighbourhoodRule
    :param verbose: verbosity, defaults to False
    :type verbose: bool, optional
    :param max_iter: maximum iteration count, defaults to 5000
    :type max_iter: int, optional
    :example:
        
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

        :param initial_solution: an initial array of crucibles.
        :type initial_solution: list[Crucible]
        :returns: the optimal solution and objective value
        :rtype: tuple[list[Crucible], float]

        :example:

            >>> import pathlib
            >>> import pandas as pd

            >>> from smelter_optimisation.neighbourhood_rule import SwapTwoPotsRule
            >>> from smelter_optimisation.solver import NextAscentSolver
            >>> from smelter_optimisation.utils import create_init_sol
            
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
                if (new_objective_value - self._current_value) > config.TOL:
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
            if np.abs(self._current_value - optimal_value) < config.TOL:
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
