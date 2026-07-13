import pytest

from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.signal import EncodingScheme, Signal, SignalKind


def make_signal(**overrides):
    base = dict(
        id=1,
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=0,
        origin="emitter_1",
        wavelength_nm=1550.0,
        correlation_id=None,
        correlation_meta=None,
        state_ref=None,
        protocol_params=(("bb84.basis", "Z"), ("bb84.bit", 1)),
        meta=(("note", "initial"),),
        timing_meta=(("emission_offset_ticks", 0),),
    )
    base.update(overrides)
    return Signal(**base)


def test_signal_preserves_transport_qstate_and_timing_metadata() -> None:
    q0 = SubsystemHandle(label="q0", kind="qubit", index=0)
    signal = make_signal(
        correlation_id=12,
        correlation_meta=(("pair", "left"),),
        state_ref=42,
        state_targets=(q0,),
        protocol_params=(("bb84.basis", "Z"),),
        meta=(("path", "alice-bob"),),
        timing_meta=(("emission_offset_ticks", 3),),
    )

    assert signal.correlation_id == 12
    assert signal.correlation_meta == (("pair", "left"),)
    assert signal.state_ref == 42
    assert signal.state_targets == (q0,)
    assert signal.protocol_params == (("bb84.basis", "Z"),)
    assert signal.meta == (("path", "alice-bob"),)
    assert signal.timing_meta == (("emission_offset_ticks", 3),)


def test_signal_hash_and_equality_include_transport_metadata() -> None:
    s1 = make_signal()
    s2 = make_signal()
    different_meta = make_signal(meta=(("different", "metadata"),))

    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert s1 != different_meta
    assert hash(s1) != hash(different_meta)


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"state_ref": "ref-1"}, TypeError, "state_ref must be int or None"),
        (
            {"correlation_id": True},
            TypeError,
            "correlation_id must be int, UUID, or None",
        ),
        (
            {"state_targets": [SubsystemHandle(label="q0")]},
            TypeError,
            "state_targets must be tuple",
        ),
        (
            {"state_targets": ("q0",)},
            TypeError,
            "state_targets must contain SubsystemHandle",
        ),
    ],
)
def test_qstate_reference_metadata_rejects_ambiguous_targets(
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        make_signal(**overrides)


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("emission_time", "0", TypeError, "emission_time must be int"),
        ("emission_time", -1, ValueError, "emission_time cannot be negative"),
        ("wavelength_nm", 0.0, ValueError, "wavelength_nm must be positive"),
        ("origin", "", ValueError, "origin must be a non-empty string"),
        ("signal_kind", "photon", TypeError, "signal_kind must be SignalKind"),
        (
            "encoding_scheme",
            SignalKind.PHOTON,
            TypeError,
            "encoding_scheme must be EncodingScheme",
        ),
    ],
)
def test_physical_signal_fields_reject_invalid_transport_metadata(
    field: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        make_signal(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_params", {"bb84.basis": "Z"}, "protocol_params must be a tuple"),
        (
            "protocol_params",
            (("bb84.basis",),),
            "protocol_params must be a tuple of",
        ),
        ("protocol_params", ((1, "Z"),), "protocol_params keys must be strings"),
        ("meta", {"path": "A-B"}, "meta must be a tuple"),
        ("meta", (("path",),), "meta must be a tuple of"),
        ("meta", ((1, "A-B"),), "meta keys must be strings"),
        ("timing_meta", {"delay_ticks": 5}, "timing_meta must be a tuple"),
        ("timing_meta", (("delay_ticks",),), "timing_meta must be a tuple of"),
        ("timing_meta", ((1, 5),), "timing_meta keys must be strings"),
    ],
)
def test_signal_metadata_channels_require_tuple_string_key_pairs(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        make_signal(**{field: value})
