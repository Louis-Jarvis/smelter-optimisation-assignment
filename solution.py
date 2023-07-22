from dataclasses import dataclass, field
from typing import List, Generator, Any
import numpy as np
import pandas as pd
import itertools
import warnings
from copy import deepcopy


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


# TODO as function
# class Neighbourhood:
class Neighbourhood:
    """Generate neighbouring solutions by swapping two pots from different crucibles.

    N(x) = {y(x, p1, p2, c1, c2) :  p1=0,...,50 & p2=0,...,50 & (a != b) & (c1 != c2) }
    - p1 is the index of the first pot
    - p2 is the index of the 2nd pot
    - c1 is the index of the first crucible
    - c2 is the index of the first

    - where x is the current solution

    y(x, p1, p2, c1, c2) = {y1, y2, ..., y17} is a neighbour of the current solution.

    Neighbourhood rule:

    yi = {
        x_{p2} if i = p1 & (c1 != c2)
        x_{p1} if i = p2 & (c1 != c2)
        x_{i} otherwise
        }
    """

    def __init__(self, x: List[Crucible]):
        self.x = x

    def __iter__(self):
        crucible_indices = set(itertools.combinations(range(17), 2))
        pot_indices = set(itertools.combinations(range(3), 2))

        x_i = deepcopy(self.x)

        for c1, c2 in crucible_indices:
            if c1 != c2:
                # TODO verbose option
                for p1, p2 in pot_indices:
                    if p1 != p2:
                        temp = x_i[c1][p1]

                        x_i[c1][p1] = x_i[c2][p2]
                        x_i[c2][p2] = temp

                        yield x_i  # (c1, c2, p1, p2)


# neighbourhood as a class that is passed to steepest ascent?


# TODO create an abstract base class for this
class NextAscent:
    max_iter = 500
    tol = 1e-6

    def __init__(self, verbose=False) -> None:
        # self.neighbours = neighbourhood
        self.verbose = verbose
        self.converged = False
        self.x_hist = []
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
            x_neighbours = Neighbourhood(x_current)

            # verbose option #TODO

            # Neighbourhood search
            for x_new in x_neighbours:
                self._num_iter += 1
                fx_new = calc_objective(x_new)
                dfx = fx_new - calc_objective(x_current)

                if dfx > self.tol:
                    x_current = x_new
                    fx_current = fx_new
                    self.x_hist.append(x_new)
                    self.fx_hist.append(fx_current)

                if self._num_iter == self.max_iter:
                    warnings.warn(
                        f"Maximum number of iterations ({self.max_iter}) reached."
                    )
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


# TODO store this in a csv, read in...
# Intitial Solution
PotAl = [
    99.79,
    99.23,
    99.64,
    99.88,
    99.55,
    99.87,
    99.55,
    99.19,
    99.76,
    99.70,
    99.26,
    99.60,
    99.05,
    99.49,
    99.69,
    99.48,
    99.60,
    99.89,
    99.39,
    99.48,
    99.77,
    99.57,
    99.48,
    99.85,
    99.09,
    99.64,
    99.71,
    99.59,
    99.14,
    99.87,
    99.38,
    99.56,
    99.32,
    99.55,
    99.61,
    99.57,
    99.75,
    99.63,
    99.17,
    99.97,
    99.74,
    99.49,
    99.75,
    99.40,
    99.72,
    99.95,
    99.31,
    99.55,
    99.29,
    99.09,
    99.20,
]

PotFe = [
    0.01,
    0.68,
    0.25,
    0.61,
    0.13,
    0.77,
    0.48,
    0.18,
    0.66,
    0.43,
    0.13,
    0.87,
    0.96,
    0.47,
    0.51,
    0.73,
    0.04,
    0.76,
    0.89,
    0.90,
    0.96,
    0.73,
    0.88,
    0.43,
    0.60,
    0.37,
    0.51,
    0.26,
    0.30,
    0.46,
    0.21,
    0.77,
    0.43,
    0.52,
    0.63,
    0.76,
    0.02,
    0.75,
    0.90,
    0.53,
    0.14,
    0.10,
    0.31,
    0.20,
    0.45,
    0.67,
    0.56,
    0.24,
    0.72,
    0.56,
    0.01,
]


# TODO add fx as an output
def create_init_sol(pot_al=PotAl, pot_fe=PotFe):
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


# TODO move these into a df

quality_df = pd.DataFrame(
    data={
        "QualityMinAl": [
            95.00,
            99.10,
            99.10,
            99.20,
            99.25,
            99.35,
            99.50,
            99.65,
            99.75,
            99.85,
            99.90,
        ],
        "QualityMaxFe": [
            5.00,
            0.81,
            0.79,
            0.79,
            0.76,
            0.72,
            0.53,
            0.50,
            0.46,
            0.33,
            0.30,
        ],
        "QualityValue": [
            10,
            21.25,
            26.95,
            36.25,
            41.53,
            44.53,
            48.71,
            52.44,
            57.35,
            68.21,
            72.56,
        ],
    }
)


def calc_crucible_value(crucible: Crucible, quality_df=quality_df, tol=1e-5):
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
