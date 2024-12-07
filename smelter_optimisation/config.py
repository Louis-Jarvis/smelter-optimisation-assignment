from pathlib import Path

import pandas as pd

INITIAL_SOL_PATH = Path("data/initial_solution.csv")
QUALITY_DATA_PATH = Path("data/quality.csv")


quality_table = pd.read_csv(QUALITY_DATA_PATH)

inital_solution = pd.read_csv(INITIAL_SOL_PATH)

TOL = 1e-6
