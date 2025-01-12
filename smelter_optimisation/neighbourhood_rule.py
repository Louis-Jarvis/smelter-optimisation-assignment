"""Classes to generate solutions adjascent to the current pot arrangement."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Generator

from numpy.typing import NDArray

from smelter_optimisation.config import NUM_CRUCIBLES, POTS_PER_CRUCIBLE

logger = logging.getLogger(__name__)


class NeighbourhoodRule(ABC):
    """Interface to generate neighbouring solutions."""

    @abstractmethod
    def generate_neighbours(self, current_solution):
        pass


class SwapTwoPotsRule(NeighbourhoodRule):
    """Generates neighbouring solutions by swapping two pots from different crucibles.

    Args:
        num_crucibles (int, optional): number of crucibles. Defaults to NUM_CRUCIBLES.
        pots_per_crucible (int, optional): number of pots in each crucible.
            Defaults to POTS_PER_CRUCIBLE.
    """

    def __init__(self, num_crucibles: int = NUM_CRUCIBLES, pots_per_crucible: int = POTS_PER_CRUCIBLE) -> None:
        self.num_crucibles = num_crucibles
        self.pots_per_crucible = pots_per_crucible

    def generate_neighbours(self, current_solution: NDArray[Any]) -> Generator[NDArray[Any]]:
        """Generate a series of neighbours around a solution by swapping two pots.

        Args:
            current_solution (NDArray[Any]): current array.

        Returns:
            Generator[NDArray[Any]]: array with two pots swapped.
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
                                logging.info(f"pot {pot_1}, crucible {crucible_1} <-> pot {pot_2}, crucible {crucible_2}") # noqa: E501
                                temp = swapped_array[crucible_1][pot_1]

                                swapped_array[crucible_1][pot_1] = swapped_array[crucible_2][pot_2]
                                swapped_array[crucible_2][pot_2] = temp

                                yield swapped_array
