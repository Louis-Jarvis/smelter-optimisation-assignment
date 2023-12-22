from abc import ABC, abstractmethod

from .crucible import Crucible
from copy import deepcopy
import itertools


class NeighbourhoodRule(ABC):
    """Interface object that contains logic to generate neighbouring solutions."""

    @abstractmethod
    def get_neighbours(x: Crucible):
        pass


class Swap2PotsRule(NeighbourhoodRule):
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
