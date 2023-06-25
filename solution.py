from dataclasses import dataclass
from typing import List, Generator, Any
import numpy as np
import pandas as pd
import itertools


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

    def __repr__(self) -> str:
        pot_num = [pot.index for pot in self.pots]
        return f"Crucible: p={pot_num}"

    def __getitem__(self, i):
        return self.pots[i]

    def __setitem__(self, i, pot: Pot):
        self.pots[i] = pot

    @property  # needed?
    def avg_Al(self):
        al_avg = np.mean([pot.Al for pot in self.pots])
        return al_avg
        # (pots[0].Al + pots[1].Al + pots[2].Al)/3

    @property
    def avg_Fe(self):
        fe_avg = np.mean([pot.Fe for pot in self.pots])
        return fe_avg
        # (pots[0].Fe + pots[1].Fe + pots[2].Fe)/3

    # # TODO quality func instead of avg al and FE
    # def get_value(self):
    #     return calc_quality(self.avg_Al, self.avg_Fe)


# TODO is this class necessary - could move the values into the solution value class?


# FIXME
def calc_objective(solution: List[Crucible]):
    return np.sum([calc_crucible_value(crucible) for crucible in solution])


def calc_quality(crucible: Crucible):
    pass  # quality calculation for each crucible


# TODO as function
# class Neighbourhood:
def gen_neighbourhood(x: List[Crucible]) -> Generator[List[Crucible], Any, None]:
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
    crucible_indices = set(itertools.combinations(range(17), 2))
    pot_indices = set(itertools.combinations(range(3), 2))
    for c1, c2 in crucible_indices:
        if c1 != c2:
            # TODO verbose option
            for p1, p2 in pot_indices:
                if p1 != p2:
                    temp = x[c1][p1]

                    x[c1][p1] = x[c2][p2]
                    x[c2][p2] = temp

                    yield x, (c1, c2, p1, p2)


# neighbourhood as a class that is passed to steepest ascent?


# TODO create an abstract base class for this
class NextAscent:
    max_iter = 15

    def __init__(self, neighbourhood_func) -> None:
        self.neighbours = neighbourhood_func
        self.converged = False
        self.x_hist = []
        self.fx_hist = []

    # TODO progress bar
    def solve(self, x0, f0):
        current_max = f0
        current_sol = x0
        self.x_hist.append(x0)
        self.fx_hist.append(f0)

        for i in self.max_iter:
            Nx = self.neighbours(current_sol)

            for x_new, (c1, c2, p1, p1) in Nx:
                dfx = self.calc_delta_fx(x0, x_new, c1, c2)

                if dfx > 0:
                    current_sol = x_new
                    self.x_hist.append(x_new)
                    self.fx_hist.append(current_max + dfx)
                else:
                    continue

            # TODO break when converged

        return

    @staticmethod  # calculate the effect of the swap on objective func
    def calc_delta_fx(x0, x_new, c1, c2):
        # since the solution is additive we can calculate the change in
        delta_1 = x0[c1].get_value() - x_new[c1].get_value()
        delta_2 = x0[c2].get_value() - x_new[c2].get_value()
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
    proportions = list(zip(pot_al, pot_fe))
    for i in range(0, 51, 3):
        cruc = Crucible(
            [
                Pot(i, proportions[i][0], proportions[i][1]),
                Pot(i + 1, proportions[i + 1][0], proportions[i + 1][1]),
                Pot(i + 2, proportions[i + 2][0], proportions[i + 2][1]),
            ]
        )
        sol.append(cruc)
        fx += calc_crucible_value(cruc)

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


def calc_crucible_value(crucible: Crucible, quality_df=quality_df, tol=1e5):
    value = 0
    # TODO vectorise this
    for i in range(len(quality_df)):
        if crucible.avg_Al >= quality_df.loc[i, "QualityMinAl"] - tol:
            if crucible.avg_Fe <= quality_df.loc[i, "QualityMaxFe"] + tol:
                value = quality_df.loc[i, "QualityValue"]

    return value
