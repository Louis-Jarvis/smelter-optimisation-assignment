import itertools
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from .crucible import Crucible

NUM_CRUCIBLES = 17
POTS_PER_CRUCIBLE = 3


class NeighbourhoodRule(ABC):
    """Interface object that contains logic to generate neighbouring solutions."""

    @abstractmethod
    def generate_neighbours(self, x: Crucible):
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

    # def __init__(self, verbose=False) -> None:
    #     self.verbose = verbose

    def generate_neighbours(self, pots_array):
        # swapped_array = deepcopy(pots_array)

        crucible_indices = itertools.combinations(range(NUM_CRUCIBLES), 2)
        pot_indices = itertools.combinations(range(POTS_PER_CRUCIBLE), 2)

        # for crucible_1, crucible_2 in crucible_indices:
        #     if crucible_1 != crucible_2:
        #         for pot_1, pot_2 in pot_indices:
        #             if pot_1 != pot_2:
        #                 swapped_array = deepcopy(pots_array)
        #                 print(
        #                     f"pot {pot_1}, crucible{crucible_1} -> pot {pot_2}, crucible {crucible_2}"
        #                 )
        #                 temp = swapped_array[crucible_1][pot_1]

        #                 swapped_array[crucible_1][pot_1] = swapped_array[crucible_2][
        #                     pot_2
        #                 ]
        #                 swapped_array[crucible_2][pot_2] = temp

        #                 # yield crucible_1, crucible_2, pot_1, pot_2
        #                 yield swapped_array

        for crucible_1 in range(0, NUM_CRUCIBLES):
            for crucible_2 in range(0, NUM_CRUCIBLES):
                if crucible_1 != crucible_2:
                    for pot_1 in range(0, POTS_PER_CRUCIBLE):
                        for pot_2 in range(0, POTS_PER_CRUCIBLE):
                            if pot_1 != pot_2:
                                swapped_array = deepcopy(pots_array)
                                print(f"pot {pot_1}, crucible{crucible_1} -> pot {pot_2}, crucible {crucible_2}")
                                temp = swapped_array[crucible_1][pot_1]

                                swapped_array[crucible_1][pot_1] = swapped_array[crucible_2][pot_2]
                                swapped_array[crucible_2][pot_2] = temp

                                # yield crucible_1, crucible_2, pot_1, pot_2
                                yield swapped_array

        # x_i = deepcopy(x)  # NEEd this due to lists being mutable

        # for c1, c2 in crucible_indices:
        #     if c1 != c2:
        #         for p1, p2 in pot_indices:
        #             if p1 != p2:
        #                 temp = x_i[c1][p1]
        #                 if self.verbose:
        #                     print(f"pot {p1}, crucible{c1} -> pot {p2}, crucible {c2}")

        #                 x_i[c1][p1] = x_i[c2][p2]
        #                 x_i[c2][p2] = temp

        #                 yield x_i


# TODO modify this
