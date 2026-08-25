"""The optical operations shipped in ``coherent_optics.py``.

Pure arithmetic, no timeline. Construction and derived properties of
``CoherentState`` are covered in ``test_coherent_state.py``, and the scalar
validators in ``tests/primitives/test_validation.py``; neither is re-tested here.

``split_50_50``, ``gaussian_temporal_overlap`` and ``interfere`` ship one commit
ahead of their first caller, the delay interferometer, so these are the whole of
their coverage until that component exists. They are tested against the
identities rather than against remembered numbers: energy conservation across
every overlap, the denominator that separates a field envelope from an intensity
one, and which port is dark.
"""

from __future__ import annotations

import cmath
from math import exp, pi, sqrt

import pytest

from simyuj.components.coherent_optics import (
    attenuated,
    click_probability,
    gaussian_temporal_overlap,
    interfere,
    phase_shifted,
    split_50_50,
)
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


# --------------------------------------------------------------------------
# split_50_50
# --------------------------------------------------------------------------


def test_split_50_50_halves_mu_keeps_the_phase_and_returns_one_object_twice() -> None:
    state = CoherentState.from_mean_photon_number(0.4, phase_rad=0.9)
    left, right = split_50_50(state)

    assert left.mean_photon_number == pytest.approx(0.2, abs=ATOL)
    assert left.mean_photon_number + right.mean_photon_number == pytest.approx(
        state.mean_photon_number,
        abs=ATOL,
    )
    assert _wrapped_delta(left.phase_rad, 0.9) == pytest.approx(0.0, abs=ATOL)

    # With vacuum on the second input the real 50:50 matrix gives (a-0)/sqrt2
    # and (a+0)/sqrt2, which are equal -- so the same frozen object comes back
    # twice. Pinned because it would otherwise surprise an identity assertion
    # that meant to test something else.
    assert left is right


def test_split_50_50_rejects_a_non_coherent_state() -> None:
    with pytest.raises(TypeError, match="state must be CoherentState"):
        split_50_50(0.5 + 0j)


# --------------------------------------------------------------------------
# gaussian_temporal_overlap
# --------------------------------------------------------------------------


def test_overlap_is_exactly_one_for_identical_aligned_modes() -> None:
    assert gaussian_temporal_overlap(
        sigma_a_s=1e-11,
        sigma_b_s=1e-11,
        delta_s=0.0,
    ) == pytest.approx(1.0, abs=ATOL)


def test_equal_width_overlap_uses_a_denominator_of_four_not_eight() -> None:
    # The field-versus-intensity envelope trap, asserted where the two answers
    # are far apart. A field sigma gives exp(-dt^2 / (4 sigma^2)); an intensity
    # sigma gives 8 in the denominator. At dt = 2 sigma that is exp(-1) against
    # exp(-0.5) -- 0.368 against 0.607, which no tolerance hides.
    sigma = 1e-11
    gamma = gaussian_temporal_overlap(
        sigma_a_s=sigma,
        sigma_b_s=sigma,
        delta_s=2.0 * sigma,
    )

    assert gamma == pytest.approx(exp(-1.0), abs=ATOL)
    assert abs(gamma - exp(-0.5)) > 0.2


def test_unequal_widths_cannot_overlap_perfectly_even_when_aligned() -> None:
    # The prefactor is the width mismatch alone, so it bites at delta = 0.
    gamma = gaussian_temporal_overlap(
        sigma_a_s=1e-11,
        sigma_b_s=2e-11,
        delta_s=0.0,
    )

    assert gamma == pytest.approx(sqrt(2.0 * 1.0 * 2.0 / 5.0), abs=ATOL)
    assert gamma < 1.0


def test_overlap_is_symmetric_in_the_widths_and_in_the_sign_of_delta() -> None:
    forward = gaussian_temporal_overlap(
        sigma_a_s=1e-11,
        sigma_b_s=3e-11,
        delta_s=4e-11,
    )

    assert forward == gaussian_temporal_overlap(
        sigma_a_s=3e-11,
        sigma_b_s=1e-11,
        delta_s=4e-11,
    )
    # delta_s is a signed centre-to-centre separation; only its magnitude
    # matters, so a caller need not order the two pulses in time.
    assert forward == gaussian_temporal_overlap(
        sigma_a_s=1e-11,
        sigma_b_s=3e-11,
        delta_s=-4e-11,
    )


def test_overlap_rejects_a_zero_width_rather_than_taking_the_discrete_limit() -> None:
    # Same rationale as test_attenuated_rejects_gain: the rejection is a
    # modelling decision, not coverage of require_positive_real. The sigma -> 0
    # limit is a different, discrete model in which overlap is an equality test
    # on ticks, and answering it from this formula would be a silent wrong turn.
    with pytest.raises(ValueError):
        gaussian_temporal_overlap(sigma_a_s=0.0, sigma_b_s=1e-11, delta_s=0.0)


# --------------------------------------------------------------------------
# interfere
# --------------------------------------------------------------------------


def test_port_zero_is_dark_and_port_one_is_bright_for_arms_in_phase() -> None:
    arm = CoherentState.from_mean_photon_number(0.2)
    dark, bright = interfere(arm, arm)

    # Never asserted as == 0.0: the value is whatever the inputs' own rounding
    # leaves, and it is 1.5e-33 rather than 0.0 as soon as one arm has been
    # through phase_shifted(pi).
    assert dark.mean_photon_number == pytest.approx(0.0, abs=ATOL)
    assert bright.mean_photon_number == pytest.approx(0.4, abs=ATOL)


def test_a_pi_phase_between_the_arms_swaps_which_port_is_dark() -> None:
    arm = CoherentState.from_mean_photon_number(0.2)
    bright, dark = interfere(arm, phase_shifted(arm, phase_rad=pi))

    assert bright.mean_photon_number == pytest.approx(0.4, abs=ATOL)
    assert dark.mean_photon_number == pytest.approx(0.0, abs=ATOL)


@pytest.mark.parametrize("overlap", [1.0, 0.5, 0.0, 0.3 + 0.4j, -1.0])
@pytest.mark.parametrize(
    ("short_mu", "long_mu", "long_phase"),
    [
        (0.2, 0.2, 0.0),
        (0.2, 0.2, pi),
        (0.35, 0.05, 1.1),
        (0.2, 0.0, 0.0),
        (0.0, 0.2, 0.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_energy_is_conserved_at_every_overlap_and_every_input(
    overlap,
    short_mu,
    long_mu,
    long_phase,
) -> None:
    # The identity that catches a beamsplitter-convention error, asserted on
    # every case exactly as the design requires. Vacuum inputs, zero overlap and
    # unequal amplitudes are values of the equation, not branches around it.
    short = CoherentState.from_mean_photon_number(short_mu)
    long_ = CoherentState.from_mean_photon_number(long_mu, phase_rad=long_phase)

    port_0, port_1 = interfere(short, long_, overlap=overlap)

    assert port_0.mean_photon_number + port_1.mean_photon_number == pytest.approx(
        short_mu + long_mu,
        abs=ATOL,
    )


def test_zero_overlap_gives_a_plain_split_with_no_interference_term() -> None:
    # Two arms that cannot interfere each split 50:50 and the ports come out
    # equal, whatever the relative phase would otherwise have done.
    short = CoherentState.from_mean_photon_number(0.2)
    long_ = CoherentState.from_mean_photon_number(0.2)

    port_0, port_1 = interfere(short, long_, overlap=0.0)

    assert port_0.mean_photon_number == pytest.approx(0.2, abs=ATOL)
    assert port_1.mean_photon_number == pytest.approx(0.2, abs=ATOL)


def test_a_vacuum_partner_kills_interference_at_any_overlap() -> None:
    # This is what makes the first and last pulse of a train ordinary values of
    # the equation: the interference term is proportional to both amplitudes.
    present = CoherentState.from_mean_photon_number(0.4)
    vacuum = CoherentState(0j)

    for overlap in (1.0, 0.5, 0.0):
        for short, long_ in ((present, vacuum), (vacuum, present)):
            port_0, port_1 = interfere(short, long_, overlap=overlap)
            assert port_0.mean_photon_number == pytest.approx(0.2, abs=ATOL)
            assert port_1.mean_photon_number == pytest.approx(0.2, abs=ATOL)


def test_an_overlap_above_unit_modulus_is_rejected() -> None:
    # Not a generic validator test: at |gamma| > 1 the orthogonal remainder goes
    # negative, the max() clamp hides it, and the two ports carry more light
    # than entered. The bound is what makes energy conservation true.
    arm = CoherentState.from_mean_photon_number(0.2)

    with pytest.raises(ValueError, match="modulus at most 1"):
        interfere(arm, arm, overlap=1.5)

    # A value a hair over one is floating-point noise from a caller's own
    # computation and is admitted.
    assert interfere(arm, arm, overlap=1.0 + 1e-15) is not None


def test_interfere_names_the_arm_that_was_not_a_coherent_state() -> None:
    arm = CoherentState.from_mean_photon_number(0.2)

    with pytest.raises(TypeError, match="short_arm must be CoherentState"):
        interfere(0.5 + 0j, arm)
    with pytest.raises(TypeError, match="long_arm must be CoherentState"):
        interfere(arm, 0.5 + 0j)


# --------------------------------------------------------------------------
# click_probability
# --------------------------------------------------------------------------


def test_click_probability_is_the_poisson_closed_form() -> None:
    # Thinning Poisson(mu) by eta gives Poisson(eta*mu), so P(n >= 1) is
    # 1 - exp(-eta*mu) exactly. Checked against the direct form rather than
    # against hand-typed constants, so the identity is what is asserted.
    for mu in (0.05, 0.2, 1.0, 7.5):
        for eta in (0.15, 0.5, 1.0):
            assert click_probability(mu, efficiency=eta) == pytest.approx(
                1.0 - exp(-eta * mu),
                abs=ATOL,
            )


def test_click_probability_is_zero_when_either_factor_is_zero() -> None:
    # Coherent vacuum is a real pulse occupying its slot, and a blind detector
    # is a real detector. Both give exactly zero with no branch.
    assert click_probability(0.0, efficiency=0.9) == 0.0
    assert click_probability(0.9, efficiency=0.0) == 0.0


def test_click_probability_saturates_to_exactly_one_in_double_precision() -> None:
    # Analytically P < 1 for every finite mu, but above eta*mu ~ 37 the gap is
    # under the double epsilon. Pinned rather than worked around: a detector
    # reading exactly 1.0 clicks without drawing from its efficiency stream, so
    # where this saturates is where a later window's stream position changes.
    assert click_probability(30.0, efficiency=1.0) < 1.0
    assert click_probability(50.0, efficiency=1.0) == 1.0


def test_click_probability_resolves_a_dark_port_without_a_special_case() -> None:
    # An analytically dark port arrives at ~1e-33 -- the exp(1j*pi) residue,
    # squared -- and total attenuation arrives at exactly 0.0. The closed form
    # has to put both at the floor on its own, because the detector downstream
    # has no branch for either.
    residue = click_probability(1.5e-33, efficiency=0.2)
    assert 0.0 <= residue < 1e-30
    assert click_probability(0.0, efficiency=0.2) == 0.0


def test_click_probability_keeps_its_digits_at_small_mu() -> None:
    # expm1, not 1.0 - exp(-x). At eta*mu = 1e-17 the naive form returns exactly
    # 0.0 and loses every significant digit to cancellation; this must not.
    tiny = click_probability(1e-17, efficiency=1.0)
    assert 1.0 - exp(-1e-17) == 0.0
    assert tiny == pytest.approx(1e-17, rel=1e-9)
