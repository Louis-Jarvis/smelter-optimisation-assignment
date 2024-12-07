"""Miscellaneous utility functions."""

import pandas as pd

from .config import NUM_POTS, POTS_PER_CRUCIBLE, QUALITY_TABLE, TOL
from .solver.models import Crucible, Pot


def calc_crucible_value(crucible: Crucible, quality_df: pd.DataFrame=QUALITY_TABLE) -> float:
    """Calculate the dollar value of an individual crucible.

    :param crucible: solution.
    :type crucible: Crucible
    :param quality_df: table mapping proportion of iron and aluminium to value,
    defaults to quality_table
    :type quality_df: pd.DataFrame, optional
    :return: the dollar value of the crucible.
    :rtype: float
    """
    value = 0
    # TODO vectorise this
    for i in range(len(quality_df) - 1, 0, -1):
        if crucible.avg_al >= quality_df.loc[i, "QualityMinAl"] - TOL:
            if crucible.avg_fe <= quality_df.loc[i, "QualityMaxFe"] + TOL:
                value = quality_df.loc[i, "QualityValue"]
                return value

    return value


def create_init_sol(initial_solution: pd.DataFrame) -> list[Crucible]:
    """Create an initial Crucible arrangement from csv of pots.

    :param initial_solution: table of pot aluminium and iron.
    :type initial_solution: pd.DataFrame
    :return: initial arrangement of pots.
    :rtype: list[Crucible]
    """
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
