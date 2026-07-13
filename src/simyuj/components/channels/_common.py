"""Shared construction and validation helpers for channel components.

The helpers in this module keep classical and quantum channel setup aligned:
they normalize scalar timing/loss inputs, derive propagation delays in
simulation ticks, and create the standard ``in``/``out`` port pair. They do not
own timeline events or payload delivery.
"""

from __future__ import annotations

from simyuj.engine.component import Component
from simyuj.primitives.units import seconds_to_ticks
from simyuj.primitives.validation import require_finite_real

from ..ports import Port, PortDirection, PortKind


def _finite_float(name: str, value: object) -> float:
    return require_finite_real(value, field_name=name, type_name="numeric")


def _probability(name: str, value: object) -> float:
    number = _finite_float(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _non_negative_int(
    name: str,
    value: object,
    *,
    type_label: str = "int",
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be {type_label}")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _resolve_delay_ticks(
    delay_ticks: int | None,
    *,
    length_m: float,
    speed_m_per_s: float,
) -> int:
    """Resolve channel propagation delay in simulation ticks.

    ``delay_ticks`` is an explicit override. When it is ``None``, the helper
    computes ``length_m / speed_m_per_s`` in seconds and converts that duration
    through the simulator unit helper. The integer conversion truncates
    fractional ticks, which matters for very short channels or custom unit
    scales.
    """
    if delay_ticks is not None:
        return delay_ticks

    delay_s = length_m / speed_m_per_s
    return int(seconds_to_ticks(delay_s))


def _create_channel_ports(
    *,
    owner: Component,
    owner_id: str,
    port_kind: PortKind,
) -> tuple[Port, Port]:
    """Create the standard channel input and output ports.

    Channels are event targets, but their ports are structural routing
    endpoints. The returned tuple is ``(in, out)`` with matching port kind and
    ``INGRESS``/``EGRESS`` directions.
    """
    return (
        Port(
            name="in",
            owner=owner,
            owner_id=owner_id,
            port_kind=port_kind,
            direction=PortDirection.INGRESS,
        ),
        Port(
            name="out",
            owner=owner,
            owner_id=owner_id,
            port_kind=port_kind,
            direction=PortDirection.EGRESS,
        ),
    )
