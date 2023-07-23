from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import itertools
import warnings
from copy import deepcopy
from abc import ABC, abstractmethod
from pathlib import Path

INITIAL_SOL_PATH = Path("data/initial_solution.csv")
QUALITY_DATA_PATH = Path("data/quality_df.csv")


# there are 51 pots
@dataclass
class Pot:
    index: int
    Al: float
    Fe: float


# there are 17 crucibles
@dataclass
class Crucible:
    pots: List[Pot]

    @property
    def avg_al(self):
        return np.mean([pot.Al for pot in self.pots])

    @property
    def avg_fe(self):
        return np.mean([pot.Fe for pot in self.pots])

    def __repr__(self) -> str:
        pot_num = [pot.index for pot in self.pots]
        return f"Crucible: p={pot_num}"

    def __getitem__(self, i):
        return self.pots[i]

    def __setitem__(self, i, pot: Pot):
        self.pots[i] = pot


def calc_quality(crucible: Crucible):
    pass  # quality calculation for each crucible


class NeighbourhoodRule(ABC):
    """Interface object that contains logic to generate neighbouring solutions."""

    @abstractmethod
    def get_neighbours(x: Crucible):
        pass


class Swap2Pots(NeighbourhoodRule):
    """Concrete implementation for generating neighbouring solutions by swapping two pots from different crucibles.

    N(x) = {y(x, p1, p2, c1, c2) :  p1=0,...,50 & p2=0,...,50 & (a != b) & (c1 != c2) }
    - p1 is the index of the first pot
    - p2 is the index of the 2nd pot
    - c1 is the index of the first crucible
    - c2 is the index of the first

    where x is the current solution

    y(x, p1, p2, c1, c2) = {y1, y2, ..., y17} is a neighbour of the current solution.

    Neighbourhood rule:

    yi = x_p1 if i = p1 & (c1 != c2)
       = x_p2 if i = p2 & (c1 != c2)
       = xi otherwise
    """

    def __init__(self, verbose=False) -> None:
        self.verbose = verbose

    def get_neighbours(self, x):
        crucible_indices = set(itertools.combinations(range(17), 2))
        pot_indices = set(itertools.combinations(range(3), 2))

        x_i = deepcopy(x)

        for c1, c2 in crucible_indices:
            if c1 != c2:
                for p1, p2 in pot_indices:
                    if p1 != p2:
                        temp = x_i[c1][p1]
                        if self.verbose:
                            print(f"pot {p1}, crucible{c1} -> pot {p2}, crucible {c2}")

                        x_i[c1][p1] = x_i[c2][p2]
                        x_i[c2][p2] = temp

                        yield x_i


# TODO create an abstract base class for this
class NextAscent:
    tol = 1e-6

    def __init__(
        self,
        neighbourhood: NeighbourhoodRule,
        verbose: bool = False,
        max_iter: int = 1000,
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
        fx_current = f0
        x_current = x0
        self.fx_hist.append(f0)

        while self._num_iter < self.max_iter:
            x_neighbours = self.neighbourhood.get_neighbours(x_current)

            # Neighbourhood search
            for x_new in x_neighbours:
                self._num_iter += 1
                fx_new = calc_objective(x_new)
                dfx = fx_new - calc_objective(x_current)

                if dfx > self.tol:
                    if self.verbose:
                        print(f"Accept Swap: current best fx: {fx_new:.4f}")
                    x_current = x_new
                    fx_current = fx_new
                    self.fx_hist.append(fx_current)

                if self._num_iter == self.max_iter:
                    warnings.warn(f"Max iterations ({self.max_iter}) reached.")
                    self._x = x_current
                    self._fx = fx_current
                    break

            # check convergence
            if np.abs(fx_new - fx_current) < self.tol:
                print("Converged!")
                self.converged = True
                self.fx_hist.append(fx_new)
                self._x = x_new
                self._fx = fx_current
                return

        return

    @property
    def solution(self):
        return self._x, self._fx

    @staticmethod  # calculate the effect of the swap on objective func
    def calc_delta_fx(x0, x_new, c1, c2):
        # FIXME this is broken
        # since the solution is additive we can calculate the change in
        delta_1 = calc_crucible_value(x_new[c1]) - calc_crucible_value(x0[c1])
        delta_2 = calc_crucible_value(x_new[c2]) - calc_crucible_value(x0[c2])
        return delta_1 + delta_2


def create_init_sol(df=pd.read_csv(INITIAL_SOL_PATH)):
    pot_al, pot_fe = df["PotAl"], df["PotFe"]

    sol = []
    fx = 0

    for i in range(0, 51, 3):
        crucible_i = Crucible(
            [
                Pot(i, pot_al[i], pot_fe[i]),
                Pot(i + 1, pot_al[i + 1], pot_fe[i + 1]),
                Pot(i + 2, pot_al[i + 2], pot_fe[i + 2]),
            ]
        )

        sol.append(crucible_i)
        fx += calc_crucible_value(crucible_i)

    return sol, fx


def calc_crucible_value(
    crucible: Crucible, quality_df=pd.read_csv(QUALITY_DATA_PATH), tol=1e-5
):
    value = 0
    # TODO vectorise this
    for i in range(len(quality_df) - 1, 0, -1):
        if crucible.avg_al >= quality_df.loc[i, "QualityMinAl"] - tol:
            if crucible.avg_fe <= quality_df.loc[i, "QualityMaxFe"] + tol:
                value = quality_df.loc[i, "QualityValue"]
                return value

    return value


def calc_objective(x):
    return np.sum([calc_crucible_value(crucible) for crucible in x])
