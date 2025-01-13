"""Miscellaneous utility functions."""

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

