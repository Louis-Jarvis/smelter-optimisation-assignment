from abc import ABC, abstractmethod
import numpy as np
import warnings
from typing import List
from copy import deepcopy

from .neighbourhood_rule import NeighbourhoodRule
from ..utils import calculate_objective_value, calc_crucible_value
from .solution import Solution
from .solution import Pot

TOL = 1e-6


# TODO implement an abstract class for this
class Solver(ABC):
    pass


class NextAscentSolver:
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
        self._x = None
        self._fx = None

    # TODO progress bar
    def run_solver(self, initial_solution: Solution):
        current_objective_value = initial_solution.objective_value
        currest_solution = deepcopy(initial_solution.pots_array)

        best_neighbour = deepcopy(initial_solution.pots_array)
        neighbour_value = initial_solution.objective_value

        self.objective_value_history.append(current_objective_value)

        # neighourhood = self.neighbourhood_rule.get_neighbours(current_pots_array)

        self._counter = 0

        while True:
            # DO SWAP

            print("new neighbourhood")
            neighbourhood = self.neighbourhood_rule.generate_neighbours(
                currest_solution
            )

            ## Find the best
            for neighbour in neighbourhood:
                self._counter += 1
                # print(neighbour, end="\n\n")

                # neighbour = doSwap(c1, c2, p1, p2)

                new_objective_value = calculate_objective_value(neighbour)
                objective_change = new_objective_value - current_objective_value

                if objective_change > TOL:
                    if self.verbose:
                        print(
                            f"Accept Swap: current best fx: {new_objective_value:.4f}"
                        )
                    best_neighbour = deepcopy(neighbour)
                    neighbour_value = new_objective_value
                    break

                if self._counter == self.max_iter:
                    break

            if np.abs(current_objective_value - neighbour_value) < TOL:
                print("Woooo ")
                return
            elif current_objective_value < neighbour_value:
                currest_solution = deepcopy(best_neighbour)
                current_objective_value = neighbour_value
            else:
                if self._counter == self.max_iter:
                    warnings.warn(f"Max iterations ({self.max_iter}) reached.")
                    return

        return  # Solution(current_objective_value, current_pots_array)

    # def _find_best_neighbour(self, neighbourhood):

    #     sol =

    #     for neighbour in neighbourhood:
    #         self._counter += 1
    #         # print(neighbour, end="\n\n")

    #         # neighbour = doSwap(c1, c2, p1, p2)

    #         new_objective_value = calculate_objective_value(neighbour)
    #         objective_change = new_objective_value - self.current_objective_value

    #         if objective_change > self.tol:
    #             if self.verbose:
    #                 print(f"Accept Swap: current best fx: {new_objective_value:.4f}")

    #             return
    #     return sol, fx

    # def _check_convergence(self, new_objective_value, current_objective):
    #     return np.abs(new_objective_value - current_objective) < self.tol

    @property
    def solution(self) -> tuple[List[Pot], float]:
        return self._x, self._fx

    @staticmethod  # calculate the effect of the swap on objective func
    def calc_delta_fx(x0, x_new, c1, c2) -> float:
        # FIXME this is broken
        # since the solution is additive we can calculate the change in
        delta_1 = calc_crucible_value(x_new[c1]) - calc_crucible_value(x0[c1])
        delta_2 = calc_crucible_value(x_new[c2]) - calc_crucible_value(x0[c2])
        return delta_1 + delta_2


# def doSwap():
#     temp = swapped_array[crucible_1][pot_1]

#     swapped_array[crucible_1][pot_1] = swapped_array[crucible_2][pot_2]
#     swapped_array[crucible_2][pot_2] = temp
