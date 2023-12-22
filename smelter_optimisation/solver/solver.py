from abc import ABC, abstractmethod
import numpy as np
import warnings

from .neighbourhood_rule import NeighbourhoodRule
from ..utils import calc_objective, calc_crucible_value


# TODO implement an abstract class for this
class Solver:
    pass


class NextAscentSolver:
    tol = 1e-6

    def __init__(
        self,
        neighbourhood: NeighbourhoodRule,
        verbose: bool = False,
        max_iter: int = 2000,
    ) -> None:
        self.neighbourhood = neighbourhood
        self.verbose = verbose
        self.max_iter = max_iter
        self.converged = False

        self.fx_hist = []

        self._num_iter = 0
        self._x = None
        self._fx = None

    # TODO progress bar
    def run_solver(self, x0, f0):
        current_objective = f0
        current_sol = x0
        self.fx_hist.append(f0)

        while self._num_iter < self.max_iter:
            x_neighbours = self.neighbourhood.get_neighbours(current_sol)

            # Neighbourhood search
            for x_new in x_neighbours:
                self._num_iter += 1
                fx_new = calc_objective(x_new)
                dfx = fx_new - calc_objective(current_sol)

                if dfx > self.tol:
                    if self.verbose:
                        print(f"Accept Swap: current best fx: {fx_new:.4f}")
                    current_sol = x_new
                    current_objective = fx_new
                    self.fx_hist.append(current_objective)

                if self._num_iter == self.max_iter:
                    warnings.warn(f"Max iterations ({self.max_iter}) reached.")
                    self._x = current_sol
                    self._fx = current_objective
                    break

            # check convergence
            if np.abs(fx_new - current_objective) < self.tol:
                print("Converged!")
                self.converged = True
                self.fx_hist.append(fx_new)
                self._x = x_new
                self._fx = current_objective
                return

        return

    @property
    def solution(self) -> tuple[Any | None, Any | None]:
        return self._x, self._fx

    @staticmethod  # calculate the effect of the swap on objective func
    def calc_delta_fx(x0, x_new, c1, c2) -> float:
        # FIXME this is broken
        # since the solution is additive we can calculate the change in
        delta_1 = calc_crucible_value(x_new[c1]) - calc_crucible_value(x0[c1])
        delta_2 = calc_crucible_value(x_new[c2]) - calc_crucible_value(x0[c2])
        return delta_1 + delta_2
