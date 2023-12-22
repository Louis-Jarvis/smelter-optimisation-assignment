from dataclasses import dataclass
from typing import List

from .pot import Pot


@dataclass
class Solution:
    objective_value: float
    solution: List[Pot]
