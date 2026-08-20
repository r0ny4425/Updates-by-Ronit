"""Construction and derived-property behaviour of ``CoherentState``."""

from __future__ import annotations

import cmath
import math

import pytest

from simyuj.primitives.coherent_state import CoherentState


class TestConstruction:
    def test_accepts_int_and_float_and_converts_to_complex(self) -> None:
        assert CoherentState(2).alpha == complex(2)
        assert CoherentState(0.5).alpha == complex(0.5)
        assert isinstance(CoherentState(2).alpha, complex)

    def test_rejects_bool(self) -> None:
        with pytest.raises(TypeError, match="alpha must be"):
            CoherentState(True)

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(TypeError, match="alpha must be"):
            CoherentState("0.5")

    @pytest.mark.parametrize(
        "alpha",
        [complex(math.nan, 0.0), complex(0.0, math.nan), complex(math.inf, 0.0)],
    )
    def test_rejects_non_finite_components(self, alpha: complex) -> None:
        with pytest.raises(ValueError, match="finite"):
            CoherentState(alpha)


class TestDerivedProperties:
    def test_mean_photon_number_is_modulus_squared(self) -> None:
        state = CoherentState(complex(0.3, 0.4))
        assert state.mean_photon_number == pytest.approx(0.25)

    def test_phase_is_arg(self) -> None:
        state = CoherentState(cmath.rect(1.0, 0.7))
        assert state.phase_rad == pytest.approx(0.7)

    def test_vacuum_is_a_valid_state_with_zero_phase(self) -> None:
        vacuum = CoherentState(0j)
        assert vacuum.mean_photon_number == 0.0
        assert vacuum.phase_rad == 0.0


class TestFromMeanPhotonNumber:
    def test_zero_phase_gives_exact_real_amplitude(self) -> None:
        state = CoherentState.from_mean_photon_number(0.25)
        assert state.alpha == complex(0.5, 0.0)
        assert state.alpha.imag == 0.0

    def test_round_trip_is_close_but_not_exact(self) -> None:
        # mu passes through sqrt on the way in. Callers must compare mu with a
        # tolerance; this pins the behaviour so nobody "fixes" it with ==.
        state = CoherentState.from_mean_photon_number(0.2)
        assert state.mean_photon_number == pytest.approx(0.2)

    def test_phase_is_carried(self) -> None:
        state = CoherentState.from_mean_photon_number(1.0, phase_rad=math.pi / 3)
        assert state.phase_rad == pytest.approx(math.pi / 3)

    def test_zero_mu_is_accepted_as_vacuum(self) -> None:
        assert CoherentState.from_mean_photon_number(0.0).alpha == 0j

    def test_no_upper_bound_on_mu(self) -> None:
        # "Weak" names the source, it is not a constraint on the type.
        assert CoherentState.from_mean_photon_number(4.0).alpha == complex(2.0, 0.0)

    def test_rejects_negative_mu(self) -> None:
        with pytest.raises(ValueError):
            CoherentState.from_mean_photon_number(-0.1)


class TestImmutability:
    def test_is_frozen(self) -> None:
        state = CoherentState(1 + 0j)
        with pytest.raises(Exception):
            state.alpha = 2 + 0j  # type: ignore[misc]

    def test_equality_is_by_amplitude(self) -> None:
        assert CoherentState(0.5 + 0j) == CoherentState(0.5)
