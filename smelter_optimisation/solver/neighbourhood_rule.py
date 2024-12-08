"""Classes to generate solutions adjascent to the current pot arrangement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Generator

from numpy.typing import NDArray

from ..config import NUM_CRUCIBLES, POTS_PER_CRUCIBLE


class NeighbourhoodRule(ABC):
    """Interface object that contains logic to generate neighbouring solutions."""

    @abstractmethod
    def generate_neighbours(self, current_solution):
        """Create neighbouring solutions according to neighbourhood rule."""
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

    def __init__(self, num_crucibles: int = NUM_CRUCIBLES, pots_per_crucible: int = POTS_PER_CRUCIBLE) -> None:
        """Initialise an instance of Swap2PotsRule.

        :param num_crucibles: number of crucibles, defaults to NUM_CRUCIBLES
        :type num_crucibles: int, optional
        :param pots_per_crucible: number of pots in each crucible, defaults to POTS_PER_CRUCIBLE
        :type pots_per_crucible: int, optional
        """
        self.num_crucibles = num_crucibles
        self.pots_per_crucible = pots_per_crucible

    def generate_neighbours(self, current_solution: NDArray[Any]) -> Generator[NDArray[Any]]:
        """Generate a series of neighbours around a solution by swapping two pots.

        :param current_solution: current array.
        :type current_solution: NDArray[Any]
        :yield: array with two pots swapped.
        :rtype: Generator[NDArray[Any]]
        """
        for crucible_1 in range(0, self.num_crucibles):
            for crucible_2 in range(0, self.num_crucibles):
                if crucible_1 < crucible_2:
                    # stop equivalent swaps being duplicated
                    # pot 0, crucible 1 <-> pot 1, crucible 0
                    # pot 1, crucible 0 <-> pot 0, crucible 1
                    for pot_1 in range(0, self.pots_per_crucible):
                        for pot_2 in range(0, self.pots_per_crucible):
                            if pot_1 < pot_2:
                                swapped_array = deepcopy(current_solution)
                                print(f"pot {pot_1}, crucible {crucible_1} <-> pot {pot_2}, crucible {crucible_2}")
                                temp = swapped_array[crucible_1][pot_1]

                                swapped_array[crucible_1][pot_1] = swapped_array[crucible_2][pot_2]
                                swapped_array[crucible_2][pot_2] = temp

                                yield swapped_array
