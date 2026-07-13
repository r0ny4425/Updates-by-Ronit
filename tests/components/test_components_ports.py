from __future__ import annotations

from dataclasses import dataclass

import pytest

from simyuj.components import Port, PortDirection, PortKind
from simyuj.engine.component import Component


@dataclass(slots=True)
class Owner(Component):
    component_id: str = "owner"

    def handle_event(self, event, timeline) -> None:
        raise NotImplementedError


def test_port_rejects_non_component_owner() -> None:
    with pytest.raises(TypeError, match="owner must be an engine Component"):
        Port(
            name="out",
            owner=object(),  # type: ignore[arg-type]
            owner_id="owner",
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.EGRESS,
        )
