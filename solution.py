from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import itertools
import warnings
from copy import deepcopy
from abc import ABC, abstractmethod
from pathlib import Path


# there are 51 pots


# there are 17 crucibles


# def calc_quality(crucible: Crucible):
#     pass  # quality calculation for each crucible


# TODO create an abstract base class for this


# def create_init_sol(df=pd.read_csv(INITIAL_SOL_PATH)):
#     pot_al, pot_fe = df["PotAl"], df["PotFe"]

#     sol = []

#     for i in range(0, 51, 3):
#         crucible_i = Crucible(
#             [
#                 Pot(i, pot_al[i], pot_fe[i]),
#                 Pot(i + 1, pot_al[i + 1], pot_fe[i + 1]),
#                 Pot(i + 2, pot_al[i + 2], pot_fe[i + 2]),
#             ]
#         )

#         sol.append(crucible_i)

#     fx = calc_objective(sol)

#     return sol, fx


# def calc_crucible_value(crucible: Crucible, quality_df=pd.read_csv(QUALITY_DATA_PATH)):
#     tol = 1e-5
#     value = 0
#     # TODO vectorise this
#     for i in range(len(quality_df) - 1, 0, -1):
#         if crucible.avg_al >= quality_df.loc[i, "QualityMinAl"] - tol:
#             if crucible.avg_fe <= quality_df.loc[i, "QualityMaxFe"] + tol:
#                 value = quality_df.loc[i, "QualityValue"]
#                 return value

#     return value


# def calc_objective(x):
#     return np.sum([calc_crucible_value(crucible) for crucible in x])
