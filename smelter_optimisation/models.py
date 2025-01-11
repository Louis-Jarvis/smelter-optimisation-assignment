"""
Logical data models for the smelting optimisiation problem.

Crucibles are composed of 3 pots - each containing a mix of Aluminium and Iron.
There are 51 pots altogether and these are poured into 17 crucibles.
Most importantly composition of each crucible determines the price that it will fetched.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Pot:
    """
    Container representing a a mixture of iron and aluminium ore.
    
    :param index: Identifier for the pot.
    :type index: int
    :param alumnium_pct: Percentage of aluminium in the ore.
    :type alumnium_pct: float
    :param iron_pct: Percentage of iron in the ore.
    :type iron_pct: float

    :example:

        >>> from smelter_optimisation import Pot
        >>> pot = Pot(index=1, alumnium_pct=25.0, iron_pct=75.0)
        >>> pot.index
        1
        >>> pot.alumnium_pct
        25.0
        >>> pot.iron_pct
        75.0
    """

    index: int
    alumnium_pct: float
    iron_pct: float


@dataclass
class Crucible:
    """
    Container representing a crucible of 3 pots.

    :param pots: A list of 3 `Pot` objects.

    :example:

        >>> from smelter_optimisation import Pot, Crucible
        >>> pot1 = Pot(index=1, alumnium_pct=25.0, iron_pct=75.0)
        >>> pot2 = Pot(index=2, alumnium_pct=30.0, iron_pct=70.0)
        >>> pot3 = Pot(index=3, alumnium_pct=20.0, iron_pct=80.0)
        >>> crucible = Crucible(pots=[pot1, pot2, pot3])
        >>> crucible.avg_al
        25.0
    """
    pots: List[Pot]

    @property
    def avg_al(self) -> float:
        """Average Alumnium % of the crucible."""
        return np.mean([pot.alumnium_pct for pot in self.pots])

    @property
    def avg_fe(self) -> float:
        """Average Iron % of the crucible."""
        return np.mean([pot.iron_pct for pot in self.pots])

    def __repr__(self) -> str:
        pot_num = [pot.index for pot in self.pots]
        return f"Crucible: p={pot_num}"

    def __getitem__(self, i) -> Pot:
        return self.pots[i]

    def __setitem__(self, i, pot: Pot) -> None:
        self.pots[i] = pot
