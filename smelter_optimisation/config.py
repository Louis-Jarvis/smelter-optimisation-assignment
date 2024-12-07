"""Configuration parameters for the model."""

from pathlib import Path

import pandas as pd

INITIAL_SOL_PATH = Path("data/initial_solution.csv")
QUALITY_DATA_PATH = Path("data/quality.csv")


QUALITY_TABLE = pd.read_csv(QUALITY_DATA_PATH)

inital_solution = pd.read_csv(INITIAL_SOL_PATH)

TOL = 1e-6

NUM_CRUCIBLES = 17
POTS_PER_CRUCIBLE = 3
NUM_POTS = 51
