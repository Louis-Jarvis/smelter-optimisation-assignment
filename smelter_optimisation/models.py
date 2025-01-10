"""Logical data models for the smelting optimisiation problem."""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Pot:
    """Container representing a a mixture of iron and aluminium ore."""

    index: int
    alumnium_pct: float
    iron_pct: float


@dataclass
class Crucible:
    """Container representing a crucible of 3 pots."""

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
