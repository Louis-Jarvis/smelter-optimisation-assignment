import pytest
from smelter_optimisation.models import Pot, Crucible  # Update this import as needed

@pytest.fixture
def pot_data():
    """Fixture to create test Pot data."""
    return Pot(index=1, alumnium_pct=25.0, iron_pct=75.0)

@pytest.fixture
def crucible_data(pot_data):
    """Fixture to create a Crucible containing 3 pots."""
    pots = [pot_data, Pot(index=2, alumnium_pct=30.0, iron_pct=70.0), Pot(index=3, alumnium_pct=35.0, iron_pct=65.0)]
    return Crucible(pots=pots)

def test_pot_initialization(pot_data):
    """Test the initialization of the Pot object."""
    assert pot_data.index == 1
    assert pot_data.alumnium_pct == 25.0
    assert pot_data.iron_pct == 75.0


def test_crucible_initialization(crucible_data):
    """Test the initialization of the Crucible object."""
    assert len(crucible_data.pots) == 3
    assert crucible_data.pots[0].index == 1
    assert crucible_data.pots[1].index == 2
    assert crucible_data.pots[2].index == 3


def test_avg_al_property(crucible_data):
    """Test the avg_al property of the Crucible class."""
    expected_avg_al = (25.0 + 30.0 + 35.0) / 3
    assert crucible_data.avg_al == expected_avg_al


def test_avg_fe_property(crucible_data):
    """Test the avg_fe property of the Crucible class."""
    expected_avg_fe = (75.0 + 70.0 + 65.0) / 3
    assert crucible_data.avg_fe == expected_avg_fe


def test_repr_method(crucible_data):
    """Test the __repr__ method of the Crucible class."""
    expected_repr = "Crucible: p=[1, 2, 3]"
    assert repr(crucible_data) == expected_repr


def test_crucible_getitem(crucible_data):
    """Test the __getitem__ method of the Crucible class."""
    assert crucible_data[0].index == 1
    assert crucible_data[1].index == 2
    assert crucible_data[2].index == 3


def test_crucible_setitem(crucible_data):
    """Test the __setitem__ method of the Crucible class."""
    new_pot = Pot(index=4, alumnium_pct=40.0, iron_pct=60.0)
    crucible_data[0] = new_pot
    assert crucible_data[0].index == 4
    assert crucible_data[0].alumnium_pct == 40.0
    assert crucible_data[0].iron_pct == 60.0
