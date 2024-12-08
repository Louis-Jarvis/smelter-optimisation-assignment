from unittest import mock

import pandas as pd
import pytest

from smelter_optimisation.solver.solver import NextAscentSolver
from smelter_optimisation.utils import create_init_sol


@pytest.fixture
def initial_solution():
    return create_init_sol(pd.read_csv("data/initial_solution.csv"))


def test_initialization():
    """Check creation of Solver"""

    mock_neighbourhood_rule = mock.MagicMock()

    solver = NextAscentSolver(neighbourhood=mock_neighbourhood_rule)
    assert solver.max_iter == 5000
    assert solver.verbose is False
    assert solver.converged is False
    assert solver.objective_value_history == []

def test_solver_converge_on_no_improvement(initial_solution):
    """Solver should converge if the objective value doesn't change."""
    with mock.patch("smelter_optimisation.solver.solver.NextAscentSolver._calculate_objective_value") as mock_calculate:
        mock_calculate.return_value = 700  # Mock the objective value to be always 700

        mock_neighbourhood_rule = mock.MagicMock()
        mock_neighbourhood_rule.generate_neighbours.return_value = initial_solution

        solver = NextAscentSolver(neighbourhood=mock_neighbourhood_rule, verbose=False)

        # Run the solver
        solver.run_solver(initial_solution)

        assert solver.converged is True, "Expected the solver to converge, but it did not."

        assert solver._num_iter > 0, "Expected the solver to perform some iterations, but it did not."

        assert (
            solver.objective_value_history[0] == 700
        ), f"Expected initial objective value to be 700, but it was {solver.objective_value_history[0]}"


def test_solver_max_iter_reached(initial_solution):
    """Solver should stop after max iterations."""
    with mock.patch("smelter_optimisation.solver.solver.NextAscentSolver._calculate_objective_value") as mock_calculate:
        max_iter = 5

        # generate 10 neighbours, none leading to an improvement
        mock_calculate.side_effect = list(range(150, 10, -10))

        mock_neighbourhood_rule = mock.MagicMock()
        mock_neighbourhood_rule.generate_neighbours.side_effect = [initial_solution] * max_iter

        solver = NextAscentSolver(neighbourhood=mock_neighbourhood_rule, max_iter=max_iter)

        with pytest.warns():
            solver.run_solver(initial_solution)

        assert mock_calculate.call_count == max_iter + 1  # extra evaluation call at the start.
        assert solver._num_iter == max_iter

@pytest.mark.filterwarnings("ignore:Max iterations (10) reached.")
def test_solver_improvement_in_objective_value(initial_solution):
    """Check solver continues to improve in objective value"""
    with mock.patch("smelter_optimisation.solver.solver.NextAscentSolver._calculate_objective_value") as mock_calculate:
        mock_calculate.side_effect = [700, 690, 730, 650, 620, 600, 780, 810, 750, 760, 730]

        # Define a mock neighbourhood rule that improves the solution
        mock_neighbourhood_rule = mock.MagicMock()
        mock_neighbourhood_rule.generate_neighbours.return_value = [initial_solution] * 10

        solver = NextAscentSolver(neighbourhood=mock_neighbourhood_rule, max_iter=10)

        solver.run_solver(initial_solution)

        _, fi = solver.solution

        assert mock_calculate.call_count == 11
        assert fi == 810
        assert solver.objective_value_history[-1] == 730

