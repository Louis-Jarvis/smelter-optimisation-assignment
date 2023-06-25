from dataclasses import dataclass
from typing import List, Generator, Any
import numpy as np
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

    @property
    def avg_Al(self):
        al_avg = np.mean([pot.Al for pot in self.pots])
        # (pots[0].Al + pots[1].Al + pots[2].Al)/3

    @property
    def avg_Fe(self):
        fe_avg = np.mean([pot.Fe for pot in self.pots])
        # (pots[0].Fe + pots[1].Fe + pots[2].Fe)/3


# TODO is this class necessary - could move the values into the solution value class?
@dataclass
class Solution:
    _sol: List[Crucible]
    fx: None

    @property
    def solution(self) -> List[Crucible]:
        return self._sol

    @solution.setter
    def solution(self, new_sol):
        pass

    def calc_objective(self):
        self.fx = np.sum([self._g(x.avg_Al, x.avg_Fe) for x in self.sol])

    def _g(self, avg_Al, avg_Fe):
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
            # crucible_1 = x[c1]
            # crucible_2 = x[c2]
            for p1, p2 in pot_indices:
                if p1 != p2:
                    temp = x[c1][p1]

                    x[c1][p1] = x[c2][p2]
                    x[c2][p2] = temp

                    yield x, (c1, c2, p1, p2)


class LocalSearch:
    pass

    # TODO progress bar
    def solve():
        pass


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


def create_init_sol(pot_al=PotAl, pot_fe=PotFe):
    sol = []

    # TODO instead get 3 at a time and make a crucible, then append to the list
    proportions = list(zip(pot_al, pot_fe))
    for i in range(0, 51, 3):
        sol.append(
            Crucible(
                [
                    Pot(i, proportions[i][0], proportions[i][1]),
                    Pot(i + 1, proportions[i + 1][0], proportions[i + 1][1]),
                    Pot(i + 2, proportions[i + 2][0], proportions[i + 2][1]),
                ]
            )
        )  # FIXM
    return sol


QualityMinAl = [
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
]
QualityMaxFe = [5.00, 0.81, 0.79, 0.79, 0.76, 0.72, 0.53, 0.50, 0.46, 0.33, 0.30]
QualityValue = [
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
]
