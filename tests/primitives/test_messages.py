import pytest

from simyuj.primitives.messages import (
    ClassicalMessage,
    DeliveryReport,
    QuantumTransitPayload,
)
from simyuj.signal import EncodingScheme, Signal, SignalKind


def make_signal() -> Signal:
    return Signal(
        id=1,
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.PHASE,
        emission_time=0,
        origin="source",
    )


def test_classical_message_preserves_control_plane_metadata() -> None:
    message = ClassicalMessage(
        sender_id="alice",
        receiver_id="bob",
        body="basis=Z",
        sent_time=10,
        session_id="s1",
        message_id="m1",
        message_type="basis.announce",
        correlation_id="corr-1",
        round_id=2,
        meta=(("basis", "Z"),),
    )

    assert hash(message)
    assert message.sender_id == "alice"
    assert message.receiver_id == "bob"
    assert message.body == "basis=Z"
    assert message.sent_time == 10
    assert message.correlation_id == "corr-1"
    assert message.round_id == 2
    assert message.meta == (("basis", "Z"),)


def test_quantum_transit_payload_preserves_original_signal_reference() -> None:
    signal = make_signal()
    payload = QuantumTransitPayload(
        sender_id="src",
        receiver_id="det",
        signal=signal,
        launched_time=0,
        session_id="q1",
        channel_id="c1",
        timing_meta=(("hop_delay", 5),),
    )

    assert hash(payload)
    assert payload.signal is signal
    assert payload.launched_time == 0
    assert payload.timing_meta == (("hop_delay", 5),)


def test_delivery_report_preserves_loss_context_for_transport_observers() -> None:
    report = DeliveryReport(
        channel_id="fiber-1",
        report_time=10,
        delivered=False,
        session_id="session-1",
        payload_id=9,
        loss_reason="attenuation",
        loss_fraction=0.2,
        meta=(("distance_km", 10),),
    )

    assert hash(report)
    assert report.delivered is False
    assert report.loss_reason == "attenuation"
    assert report.loss_fraction == pytest.approx(0.2)
    assert report.meta == (("distance_km", 10),)


def test_delivery_report_accepts_integral_loss_fraction() -> None:
    report = DeliveryReport(
        channel_id="fiber-1",
        report_time=10,
        delivered=False,
        loss_fraction=1,
    )

    assert report.loss_fraction == 1


def test_classical_message_rejects_invalid_body_type() -> None:
    with pytest.raises(TypeError, match="body must be str or bytes"):
        ClassicalMessage(
            sender_id="alice",
            receiver_id="bob",
            body=1,  # type: ignore[arg-type]
            sent_time=0,
        )


def test_classical_message_rejects_bool_sent_time() -> None:
    with pytest.raises(TypeError, match="sent_time must be int"):
        ClassicalMessage(
            sender_id="alice",
            receiver_id="bob",
            body="basis=Z",
            sent_time=True,  # type: ignore[arg-type]
        )


def test_quantum_transit_payload_rejects_invalid_signal() -> None:
    with pytest.raises(TypeError, match="signal must be Signal"):
        QuantumTransitPayload(
            sender_id="src",
            receiver_id="det",
            signal="invalid",  # type: ignore[arg-type]
            launched_time=0,
        )


def test_quantum_transit_payload_rejects_bool_launched_time() -> None:
    with pytest.raises(TypeError, match="launched_time must be int"):
        QuantumTransitPayload(
            sender_id="src",
            receiver_id="det",
            signal=make_signal(),
            launched_time=True,  # type: ignore[arg-type]
        )


def test_classical_message_rejects_unhashable_meta_values() -> None:
    with pytest.raises(TypeError, match="meta values must be hashable"):
        ClassicalMessage(
            sender_id="alice",
            receiver_id="bob",
            body="basis=Z",
            sent_time=0,
            meta=(("bad", []),),
        )


def test_delivery_report_rejects_bool_report_time() -> None:
    with pytest.raises(TypeError, match="report_time must be int"):
        DeliveryReport(
            channel_id="fiber-1",
            report_time=True,  # type: ignore[arg-type]
            delivered=False,
        )


def test_delivery_report_rejects_bool_loss_fraction() -> None:
    with pytest.raises(TypeError, match="loss_fraction must be float"):
        DeliveryReport(
            channel_id="fiber-1",
            report_time=10,
            delivered=False,
            loss_fraction=True,  # type: ignore[arg-type]
        )


def test_delivery_report_rejects_out_of_range_loss_fraction() -> None:
    with pytest.raises(ValueError, match="loss_fraction must be in \\[0.0, 1.0\\]"):
        DeliveryReport(
            channel_id="fiber-1",
            report_time=10,
            delivered=False,
            loss_fraction=1.2,
        )
