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

def test_solver_improvement_in_objective_value(initial_solution):
    """Check solver handles improvement in objective value"""
    with mock.patch("smelter_optimisation.solver.solver.NextAscentSolver._calculate_objective_value") as mock_calculate:
        initial_value = 700
        improved_value = initial_value + 10
        mock_calculate.return_value = improved_value

        # Define a mock neighbourhood rule that improves the solution
        ImprovedNeighbourhoodRule = mock.MagicMock()
        ImprovedNeighbourhoodRule.generate_neighbours.return_value = [initial_solution]

        solver = NextAscentSolver(neighbourhood=ImprovedNeighbourhoodRule())

        solver.run_solver(initial_solution)

        assert solver.objective_value_history[-1] == improved_value
