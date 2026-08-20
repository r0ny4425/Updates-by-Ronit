"""The three optical ``Signal`` fields, and the ``_derived`` copy guarantee.

The guard tests in ``TestDerivedCompleteness`` are the reason ``_derived``
exists. ``Signal`` is ``slots=True``, so a field a hand-written copy forgets to
set is left *unset* and raises ``AttributeError`` on first read -- on the far
side of a channel, far from the edit that caused it. These tests assert the
property that makes that impossible, not merely that today's fields survive.
"""

from __future__ import annotations

import math
from dataclasses import fields

import pytest

from simyuj.primitives.coherent_state import CoherentState
from simyuj.signal import EncodingScheme, Signal, SignalKind
from simyuj.signal.signal import _KEEP, _SIGNAL_FIELD_NAMES

H = (1.0 + 0j, 0j)
D = (complex(2**-0.5), complex(2**-0.5))


def _signal(**overrides: object) -> Signal:
    base: dict[str, object] = {
        "id": "alice:pulse:1",
        "signal_kind": SignalKind.PULSE,
        "encoding_scheme": EncodingScheme.PHASE,
        "emission_time": 7,
        "origin": "alice",
    }
    base.update(overrides)
    return Signal(**base)  # type: ignore[arg-type]


class TestDefaults:
    def test_all_three_default_to_none(self) -> None:
        signal = _signal()
        assert signal.coherent_state is None
        assert signal.temporal_mode_sigma_s is None
        assert signal.polarization is None

    def test_optical_fields_are_appended_last(self) -> None:
        # Appended rather than grouped with wavelength_nm, so that "does any
        # call site construct Signal positionally?" stays unanswerable.
        assert _SIGNAL_FIELD_NAMES[-3:] == (
            "coherent_state",
            "temporal_mode_sigma_s",
            "polarization",
        )


class TestCoherentStateField:
    def test_carries_the_state(self) -> None:
        state = CoherentState.from_mean_photon_number(0.2)
        assert _signal(coherent_state=state).coherent_state is state

    def test_rejects_a_bare_complex(self) -> None:
        with pytest.raises(TypeError, match="coherent_state must be CoherentState"):
            _signal(coherent_state=0.5 + 0j)


class TestTemporalModeSigma:
    def test_accepts_positive_seconds(self) -> None:
        assert _signal(temporal_mode_sigma_s=1e-11).temporal_mode_sigma_s == 1e-11

    @pytest.mark.parametrize("bad", [0.0, -1e-11])
    def test_rejects_non_positive(self, bad: float) -> None:
        with pytest.raises(ValueError):
            _signal(temporal_mode_sigma_s=bad)

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(TypeError):
            _signal(temporal_mode_sigma_s="1e-11")

    def test_is_not_quantized_to_ticks(self) -> None:
        # A sub-picosecond width must survive. seconds_to_ticks would round it
        # to zero and make any overlap computed from it meaningless.
        signal = _signal(temporal_mode_sigma_s=4e-13)
        assert signal.temporal_mode_sigma_s == 4e-13


class TestPolarization:
    def test_accepts_normalized_real_components(self) -> None:
        assert _signal(polarization=H).polarization == (1 + 0j, 0j)

    def test_converts_components_to_complex(self) -> None:
        polarization = _signal(polarization=(1.0, 0.0)).polarization
        assert polarization is not None
        assert all(isinstance(component, complex) for component in polarization)

    def test_accepts_a_diagonal_state(self) -> None:
        assert _signal(polarization=D).polarization == D

    def test_rejects_unnormalized(self) -> None:
        with pytest.raises(ValueError, match="normalized"):
            _signal(polarization=(1.0 + 0j, 1.0 + 0j))

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="exactly two"):
            _signal(polarization=(1.0 + 0j,))

    def test_rejects_bool_components(self) -> None:
        with pytest.raises(TypeError):
            _signal(polarization=(True, False))


class TestValidationFlagFastPath:
    def test_skips_the_new_checks(self) -> None:
        # The emission hot path builds trusted signals with validation off; the
        # new fields must not reintroduce a cost there.
        signal = _signal(
            polarization=(9.0 + 0j, 9.0 + 0j),
            temporal_mode_sigma_s=-1.0,
            validation_flag=False,
        )
        assert signal.polarization == (9.0 + 0j, 9.0 + 0j)


class TestDerivedCompleteness:
    def test_field_name_tuple_matches_the_dataclass(self) -> None:
        # If this fails, a field was added to Signal and _derived no longer
        # knows about it. That is the whole guarantee.
        assert _SIGNAL_FIELD_NAMES == tuple(f.name for f in fields(Signal))

    def test_every_field_survives_a_derive(self) -> None:
        # Build a signal with a distinct non-default value in every field, copy
        # it, and require the copy to be equal. Enumerating from the dataclass
        # means a future field is covered without editing this test.
        distinct = _signal(
            wavelength_nm=1310.0,
            correlation_id=11,
            correlation_meta=("pair",),
            state_ref=3,
            protocol_params=(("dps.phase_index", 1),),
            meta=(("source_device_id", "alice"),),
            timing_meta=(("emission_slot_tick", 7),),
            coherent_state=CoherentState.from_mean_photon_number(0.2),
            temporal_mode_sigma_s=1e-11,
            polarization=D,
        )
        copy = distinct._derived()

        for name in _SIGNAL_FIELD_NAMES:
            assert getattr(copy, name) == getattr(distinct, name), name
        assert copy == distinct

    def test_with_metadata_preserves_the_optical_fields(self) -> None:
        # The one production caller of _derived goes through _with_metadata.
        state = CoherentState.from_mean_photon_number(0.2)
        signal = _signal(
            coherent_state=state,
            temporal_mode_sigma_s=1e-11,
            polarization=H,
        )
        annotated = signal._with_metadata(
            meta=(("quantum_channel_id", "fiber"),),
            timing_meta=(("channel_arrival_time", 12),),
        )

        assert annotated.coherent_state is state
        assert annotated.temporal_mode_sigma_s == 1e-11
        assert annotated.polarization == H
        assert annotated.meta == (("quantum_channel_id", "fiber"),)
        assert annotated.timing_meta == (("channel_arrival_time", 12),)


class TestDerivedReplacement:
    def test_replaces_only_the_named_field(self) -> None:
        signal = _signal(coherent_state=CoherentState.from_mean_photon_number(0.2))
        stronger = CoherentState.from_mean_photon_number(0.5)
        derived = signal._derived(coherent_state=stronger)

        assert derived.coherent_state is stronger
        assert derived.id == signal.id
        assert derived.emission_time == signal.emission_time

    def test_none_clears_an_amplitude(self) -> None:
        # None is a legal value, so passing it must clear rather than mean
        # "no change". This is what the _KEEP sentinel exists to disambiguate.
        signal = _signal(coherent_state=CoherentState.from_mean_photon_number(0.2))
        assert signal._derived(coherent_state=None).coherent_state is None

    def test_keep_sentinel_preserves_an_amplitude(self) -> None:
        state = CoherentState.from_mean_photon_number(0.2)
        signal = _signal(coherent_state=state)
        assert signal._derived(coherent_state=_KEEP).coherent_state is state

    def test_does_not_revalidate(self) -> None:
        # _derived is an internal transform on an already-validated signal.
        signal = _signal()
        assert signal._derived(temporal_mode_sigma_s=-1.0).temporal_mode_sigma_s == -1.0

    def test_rejects_an_unknown_field_name(self) -> None:
        with pytest.raises(TypeError, match="unknown Signal field"):
            _signal()._derived(amplitude=1.0)

    def test_returns_a_new_object(self) -> None:
        signal = _signal()
        assert signal._derived() is not signal


class TestOpticalAndQstatePayloadsCoexist:
    def test_a_signal_may_carry_both_today(self) -> None:
        # Signal does not police this combination; the channel rejects it at
        # event time, because only a transport component knows what it was
        # handed. Pinned so the policy stays where it was put.
        signal = _signal(
            state_ref=3,
            coherent_state=CoherentState.from_mean_photon_number(0.2),
        )
        assert signal.state_ref == 3
        assert signal.coherent_state is not None


def test_polarization_norm_tolerance_is_tight_enough_to_catch_real_errors() -> None:
    # 1/sqrt(2) rounded to 3 places is normalized to ~1e-4, which must fail.
    with pytest.raises(ValueError, match="normalized"):
        _signal(polarization=(0.707 + 0j, 0.707 + 0j))


def test_polarization_accepts_floating_point_normalization_noise() -> None:
    # ... but the exact float 1/sqrt(2) pair must pass.
    component = complex(1.0 / math.sqrt(2.0))
    assert _signal(polarization=(component, component)).polarization is not None
