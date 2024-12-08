"""Different heuristic solvers for the smelter optimisation solver."""

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .. import config
from ..utils import calc_crucible_value
from .models import Pot, Crucible
from .neighbourhood_rule import NeighbourhoodRule


def _calc_delta_fx(x0, x_new, c1, c2) -> float:
    # FIXME this is broken
    # since the solution is additive we can calculate the change in
    delta_1 = calc_crucible_value(x_new[c1]) - calc_crucible_value(x0[c1])
    delta_2 = calc_crucible_value(x_new[c2]) - calc_crucible_value(x0[c2])
    return delta_1 + delta_2


class SmeltingOptimisationSolver(ABC):
    """Smelter optimisation problem solver heuristic."""

    @abstractmethod
    def run_solver(self, initial_solution: list[Crucible]):
        """Generate solution."""
        pass

    @property
    @abstractmethod
    def solution(self):
        """Best solution found."""
        pass

    @abstractmethod
    def plot_objective(self):
        """Plot the objective function against the number of function evaluations."""
        pass

    @abstractmethod
    def _calculate_objective_value(self, x):
        pass


class NextAscentSolver(SmeltingOptimisationSolver):
    """Solver utilising a Next Ascent Greedy Heuristic."""

    def __init__(
        self,
        neighbourhood: NeighbourhoodRule,
        verbose: bool = False,
        max_iter: int = 5000,
    ) -> None:
        """Initialise Next Ascent Solver.

        :param neighbourhood: neighborhood rule that defines neighbours
        adjacent to the current pot.
        :type neighbourhood: NeighbourhoodRule
        :param verbose: verbosity, defaults to False
        :type verbose: bool, optional
        :param max_iter: maximum iteration count, defaults to 5000
        :type max_iter: int, optional
        """
        self.neighbourhood_rule = neighbourhood
        self.verbose = verbose
        self.max_iter = max_iter
        self.converged = False

        self.objective_value_history = []

        self._num_iter = 0
        self._x = None
        self._fx = None

    # TODO progress bar
    def run_solver(self, initial_solution):
        """Generate solution."""
        current_objective_value = self._calculate_objective_value(initial_solution)
        current_solution = deepcopy(initial_solution)

        best_neighbour = deepcopy(initial_solution)
        neighbour_value = current_objective_value

        self.objective_value_history.append(current_objective_value)

        while True:

            print("new neighbourhood")  # TODO use logger
            neighbourhood = self.neighbourhood_rule.generate_neighbours(current_solution)

            ## Find the best
            for neighbour in neighbourhood:
                self._num_iter += 1

                # evaluate neighbour
                new_objective_value = self._calculate_objective_value(neighbour)
                objective_change = new_objective_value - current_objective_value

                self.objective_value_history.append(new_objective_value)

                # check for improvement
                if objective_change > config.TOL:
                    if self.verbose:
                        print(f"Accept Swap: current best fx: {new_objective_value:.4f}")
                    best_neighbour = deepcopy(neighbour)

                    break

                if self._num_iter == self.max_iter:
                    warnings.warn(f"Max iterations ({self.max_iter}) reached.", stacklevel=1)
                    return

            if np.abs(current_objective_value - best_neighbour) < config.TOL:
                self.converged = True
                print("Woooo ")
                return
            
            if current_objective_value < neighbour_value:
                current_solution = deepcopy(best_neighbour)
                current_objective_value = neighbour_value

    def _calculate_objective_value(self, x):
        return np.sum([calc_crucible_value(crucible) for crucible in x])

    def plot_objective(self):
        """Plot the objective function against the number of function evaluations."""
        _, ax = plt.subplots()

        x = np.arange(0, len(self.objective_value_history))
        ax.plot(x, self.objective_value_history)

        ax.set(xlabel="Function Evaluations", ylabel="Objective Function Value f(x)")

        plt.show()

    @property
    def solution(self) -> tuple[List[Pot], float]:
        """Best solution found."""
        return self._x, self._fx

