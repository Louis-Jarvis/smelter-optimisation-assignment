from dataclasses import dataclass
from typing import List

import numpy as np

from .pot import Pot


@dataclass
class Crucible:
    pots: List[Pot]

    @property
    def avg_al(self) -> float:
        return np.mean([pot.Al for pot in self.pots])

    @property
    def avg_fe(self) -> float:
        return np.mean([pot.Fe for pot in self.pots])

    def __repr__(self) -> str:
        pot_num = [pot.index for pot in self.pots]
        return f"Crucible: p={pot_num}"

    def __getitem__(self, i) -> Pot:
        return self.pots[i]

    def __setitem__(self, i, pot: Pot) -> None:
        self.pots[i] = pot
