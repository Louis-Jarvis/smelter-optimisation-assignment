"""Utility functions for smelter optimization operations.

This module provides essential utility functions including:

- Loading and handling pot and crucible data from CSV files
- Calculating crucible values based on quality thresholds
- Converting between different data representations
- Displaying a pretty print of the solution

The module works with Pot and Crucible objects, handling their aluminum and iron concentrations
to support optimisation of the smelting process.
"""

from importlib import resources

import pandas as pd

from smelter_optimisation.config import NUM_POTS, POTS_PER_CRUCIBLE, TOL
from smelter_optimisation.models import Crucible, Pot


def _get_data_path(filename: str) -> str:
    """Get the path to a data file in the package."""
    with resources.path('smelter_optimisation.data', filename) as path:
        return str(path)

quality_df = pd.read_csv(_get_data_path("quality.csv"))

def calc_crucible_value(crucible: Crucible) -> float:
    """Calculate the dollar value of an individual crucible.

    Args:
        crucible (Crucible): solution.

    Returns:
        float: the dollar value of the crucible.

    Examples:
        >>> from smelter_optimisation.models import Crucible, Pot # doctest: +SKIP
        >>> from smelter_optimisation.utils import calc_crucible_value # doctest: +SKIP
        >>> crucible = Crucible([Pot(0, 0.1, 0.2), Pot(1, 0.3, 0.4), Pot(2, 0.5, 0.6)]) # doctest: +SKIP
        >>> calc_crucible_value(crucible) # doctest: +SKIP
        100
    """
    value = 0
    # TODO vectorise this
    for i in range(len(quality_df) - 1, 0, -1):
        if crucible.avg_al >= quality_df.loc[i, "QualityMinAl"] - TOL:
            if crucible.avg_fe <= quality_df.loc[i, "QualityMaxFe"] + TOL:
                value = quality_df.loc[i, "QualityValue"]
                return value

    return value


def load_initial_solution() -> list[Crucible]:
    """Create an initial Crucible arrangement from csv of pots.

    Returns:
        list[Crucible]: initial arrangement of pots.
    """
    initial_solution = pd.read_csv(_get_data_path("initial_solution.csv"))
    pot_al, pot_fe = initial_solution["PotAl"], initial_solution["PotFe"]

    sol = []

    for i in range(0, NUM_POTS, POTS_PER_CRUCIBLE):
        crucible_i = Crucible(
            [
                Pot(i, pot_al[i], pot_fe[i]),
                Pot(i + 1, pot_al[i + 1], pot_fe[i + 1]),
                Pot(i + 2, pot_al[i + 2], pot_fe[i + 2]),
            ]
        )

        sol.append(crucible_i)
    return sol

def print_solution(solution: list[Crucible]) -> None:
    """Pretty print the solution to the console.

    Args:
        solution (list[Crucible]): the solution to print.

    Examples:
        >>> from smelter_optimisation.utils import print_solution
        >>> from smelter_optimisation.utils import load_initial_solution
        >>> print_solution(load_initial_solution()) # doctest: +SKIP
        Crucible: p=[0, 1, 2]
        Crucible: p=[3, 4, 5]
        Crucible: p=[6, 7, 8]
        ...
        Crucible: p=[45, 46, 47]
        Crucible: p=[48, 49, 50]

    """
    for crucible in solution:
        print(crucible)
