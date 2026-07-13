import pytest

import simyuj.primitives.units as primitive_units


def test_simulation_ticks_use_picosecond_physical_convention() -> None:
    assert primitive_units.DEFAULT_TICK_PS == 1
    assert primitive_units.DEFAULT_TICK_SECONDS == 1e-12

    assert primitive_units.seconds_to_ps(1.25e-9) == 1250
    assert primitive_units.seconds_to_ticks(1.25e-9) == 1250
    assert primitive_units.ticks_to_seconds(1250) == pytest.approx(1.25e-9)
    assert primitive_units.ps_to_seconds(1250) == pytest.approx(1.25e-9)


def test_time_conversions_round_to_discrete_transport_units() -> None:
    assert primitive_units.seconds_to_ps(2.5e-12) == 2
    assert primitive_units.seconds_to_ns(1.25e-9) == 1
    assert primitive_units.seconds_to_us(2.5e-6) == 2
    assert primitive_units.seconds_to_ms(2.5e-3) == 2


@pytest.mark.parametrize("seconds", [-1e-12, -1.0])
def test_negative_physical_durations_are_rejected(seconds: float) -> None:
    with pytest.raises(ValueError, match="seconds cannot be negative"):
        primitive_units.seconds_to_ps(seconds)
    with pytest.raises(ValueError, match="seconds cannot be negative"):
        primitive_units.seconds_to_ns(seconds)
    with pytest.raises(ValueError, match="seconds cannot be negative"):
        primitive_units.seconds_to_us(seconds)
    with pytest.raises(ValueError, match="seconds cannot be negative"):
        primitive_units.seconds_to_ms(seconds)
    with pytest.raises(ValueError, match="seconds cannot be negative"):
        primitive_units.seconds_to_ticks(seconds)


@pytest.mark.parametrize("ps", [-1, -10])
def test_negative_simulation_ticks_are_rejected(ps: int) -> None:
    with pytest.raises(ValueError, match="picoseconds cannot be negative"):
        primitive_units.ps_to_seconds(ps)
    with pytest.raises(ValueError, match="picoseconds cannot be negative"):
        primitive_units.ticks_to_seconds(ps)
