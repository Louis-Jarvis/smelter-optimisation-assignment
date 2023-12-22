import pandas as pd
import numpy as np

from .config import quality_table
from .solver.crucible import Crucible
from .solver.pot import Pot


def calc_crucible_value(crucible: Crucible, quality_df=quality_table):
    tol = 1e-5
    value = 0
    # TODO vectorise this
    for i in range(len(quality_df) - 1, 0, -1):
        if crucible.avg_al >= quality_df.loc[i, "QualityMinAl"] - tol:
            if crucible.avg_fe <= quality_df.loc[i, "QualityMaxFe"] + tol:
                value = quality_df.loc[i, "QualityValue"]
                return value

    return value


def create_init_sol(initial_solution_df: pd.DataFrame):
    pot_al, pot_fe = initial_solution_df["PotAl"], initial_solution_df["PotFe"]

    sol = []

    for i in range(0, 51, 3):
        crucible_i = Crucible(
            [
                Pot(i, pot_al[i], pot_fe[i]),
                Pot(i + 1, pot_al[i + 1], pot_fe[i + 1]),
                Pot(i + 2, pot_al[i + 2], pot_fe[i + 2]),
            ]
        )

        sol.append(crucible_i)

    fx = calc_objective(sol)

    return sol, fx


def calc_objective(x):
    return np.sum([calc_crucible_value(crucible) for crucible in x])
