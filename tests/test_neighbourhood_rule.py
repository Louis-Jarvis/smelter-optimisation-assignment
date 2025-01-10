import pytest
import numpy as np

from smelter_optimisation.neighbourhood_rule import Swap2PotsRule

NUM_CRUCIBLES = 2
POTS_PER_CRUCIBLE = 3

@pytest.fixture
def setup_rule():
    """Fixture to set up the Swap2PotsRule."""
    return Swap2PotsRule(num_crucibles=NUM_CRUCIBLES, pots_per_crucible=POTS_PER_CRUCIBLE)

#TODO use hypothesis here
def test_generate_neighbours_size(setup_rule):
    """Test the number of neighbours generated for a given solution."""
    current_solution = np.zeros((NUM_CRUCIBLES, POTS_PER_CRUCIBLE))
    neighbours = list(setup_rule.generate_neighbours(current_solution))

    # The number of possible swaps should be (NUM_CRUCIBLES choose 2) * (POTS_PER_CRUCIBLE choose 2)
    expected_number_of_neighbours = (NUM_CRUCIBLES * (NUM_CRUCIBLES - 1) // 2) * (POTS_PER_CRUCIBLE * (POTS_PER_CRUCIBLE - 1) // 2)
    
    # Assert that the generated neighbours match the expected count
    assert len(neighbours) == expected_number_of_neighbours

def test_generate_neighbours_structure(setup_rule):
    """Test the structure of the generated neighbours."""
    current_solution = np.zeros((NUM_CRUCIBLES, POTS_PER_CRUCIBLE))

    # Generate neighbours
    neighbours = list(setup_rule.generate_neighbours(current_solution))

    # Ensure all neighbours have the correct shape
    for neighbour in neighbours:
        assert neighbour.shape == (NUM_CRUCIBLES, POTS_PER_CRUCIBLE)

def test_generate_neighbours_no_same_swap(setup_rule):
    """Ensure that the same pot is not swapped with itself."""
    current_solution = np.zeros((NUM_CRUCIBLES, POTS_PER_CRUCIBLE))
    
    # Generate neighbours
    neighbours = list(setup_rule.generate_neighbours(current_solution))
    
    # Check that no swap happens with the same pot in the same crucible
    for neighbour in neighbours:
        for crucible_1 in range(NUM_CRUCIBLES):
            for crucible_2 in range(NUM_CRUCIBLES):
                for pot_1 in range(POTS_PER_CRUCIBLE):
                    for pot_2 in range(POTS_PER_CRUCIBLE):
                        if crucible_1 == crucible_2 and pot_1 == pot_2:
                            assert current_solution[crucible_1][pot_1] == neighbour[crucible_1][pot_1]

def test_generate_neighbours_swapped_correctly(setup_rule):
    """Test that the pots are swapped correctly."""
    current_solution = np.zeros((NUM_CRUCIBLES, POTS_PER_CRUCIBLE))
    
    # Generate neighbours
    neighbours = list(setup_rule.generate_neighbours(current_solution))

    for neighbour in neighbours:
        # Find swapped pots by checking differences between current_solution and neighbour
        for crucible_1 in range(NUM_CRUCIBLES):
            for crucible_2 in range(NUM_CRUCIBLES):
                if crucible_1 != crucible_2:
                    for pot_1 in range(POTS_PER_CRUCIBLE):
                        for pot_2 in range(POTS_PER_CRUCIBLE):
                            if pot_1 != pot_2:
                                if current_solution[crucible_1][pot_1] == neighbour[crucible_2][pot_2] and \
                                   current_solution[crucible_2][pot_2] == neighbour[crucible_1][pot_1]:
                                    continue
                                else:
                                    pytest.fail("Pots were not swapped correctly.")
