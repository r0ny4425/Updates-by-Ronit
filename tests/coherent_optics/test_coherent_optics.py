"""The two optical operations that have a caller in ``src/``.

Pure arithmetic, no timeline. Construction and derived properties of
``CoherentState`` are covered in ``test_coherent_state.py``, and the scalar
validators in ``tests/primitives/test_validation.py``; neither is re-tested here.
"""

from __future__ import annotations

import cmath
from math import pi, sqrt

import pytest

from simyuj.components.coherent_optics import attenuated, phase_shifted
from simyuj.primitives.coherent_state import CoherentState

ATOL = 1e-12


def _wrapped_delta(a: float, b: float) -> float:
    """Signed difference between two phases, wrapped into ``(-pi, pi]``."""
    return (a - b + pi) % (2.0 * pi) - pi


def test_attenuated_scales_mu_by_eta_and_alpha_by_its_root() -> None:
    # The convention that matters: mu -> eta*mu, not sqrt(eta)*mu. Confusing
    # the two at 25 km of standard fibre is 0.2 against 0.45.
    state = CoherentState.from_mean_photon_number(0.4)
    out = attenuated(state, power_transmission=0.25)

    assert out.mean_photon_number == pytest.approx(0.1, abs=ATOL)
    assert abs(out.alpha) == pytest.approx(sqrt(0.4) * 0.5, abs=ATOL)


def test_attenuated_leaves_the_phase_untouched() -> None:
    # Attenuation is a real scaling. The phase has to survive it exactly, or
    # nothing downstream can interfere two attenuated pulses.
    state = CoherentState.from_mean_photon_number(0.3, phase_rad=1.234)
    out = attenuated(state, power_transmission=0.05)

    assert out.phase_rad == state.phase_rad


def test_attenuated_boundaries_are_vacuum_and_identity() -> None:
    state = CoherentState.from_mean_photon_number(0.2, phase_rad=0.6)

    # Total attenuation delivers the coherent vacuum -- a real optical state
    # that still occupies its slot, not an absent or dropped pulse.
    assert attenuated(state, power_transmission=0.0).alpha == 0j
    assert attenuated(state, power_transmission=1.0).mean_photon_number == (
        pytest.approx(0.2, abs=ATOL)
    )


def test_phase_shifted_preserves_modulus_and_adds_the_phase() -> None:
    state = CoherentState.from_mean_photon_number(0.25, phase_rad=0.5)
    out = phase_shifted(state, phase_rad=-1.25)

    assert out.mean_photon_number == pytest.approx(0.25, abs=ATOL)
    assert _wrapped_delta(out.phase_rad, 0.5 - 1.25) == pytest.approx(0.0, abs=ATOL)


def test_phase_shift_of_pi_is_not_exactly_the_negated_amplitude() -> None:
    # exp(1j*pi) is (-1+1.2246e-16j) and that residue is deliberately not
    # special-cased: a hidden branch for pi would make exactness look
    # guaranteed, and a dark port would then rest on an implementation
    # accident. If this test starts failing, someone "fixed" it.
    state = CoherentState(complex(0.5, 0.0))
    out = phase_shifted(state, phase_rad=pi)

    assert out.alpha != -state.alpha
    assert out.alpha == pytest.approx(-state.alpha, abs=1e-15)


def test_attenuation_and_phase_shift_commute_analytically_not_bit_exactly() -> None:
    # Complex multiplication is not associative in floating point. An earlier
    # draft of the design document claimed exact commutation and was wrong.
    state = CoherentState.from_mean_photon_number(0.37, phase_rad=0.91)

    first = phase_shifted(attenuated(state, power_transmission=0.3), phase_rad=2.1)
    second = attenuated(phase_shifted(state, phase_rad=2.1), power_transmission=0.3)

    assert first.alpha == pytest.approx(second.alpha, abs=1e-15)
    assert cmath.isclose(first.alpha, second.alpha, rel_tol=1e-12)


def test_attenuated_rejects_gain() -> None:
    # Optical gain is not modelled anywhere in this simulator. The rejection is
    # a physics decision, not a validator test -- require_probability's own
    # coverage lives in tests/primitives/test_validation.py.
    with pytest.raises(ValueError):
        attenuated(CoherentState(0.5 + 0j), power_transmission=1.5)


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    [
        (attenuated, {"power_transmission": 0.5}),
        (phase_shifted, {"phase_rad": 0.5}),
    ],
    ids=["attenuated", "phase_shifted"],
)
def test_operations_reject_a_non_coherent_state(operation, kwargs) -> None:
    with pytest.raises(TypeError, match="state must be CoherentState"):
        operation(0.5 + 0j, **kwargs)
