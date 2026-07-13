from __future__ import annotations

import pytest

from simyuj.components.ports import PortKind
from simyuj.network.link import NetworkLink


def test_network_link_records_physical_topology_metadata() -> None:
    channel = object()

    link = NetworkLink(
        link_id="link1",
        source_node_id="alice",
        target_node_id="bob",
        port_kind=PortKind.CLASSICAL,
        transport=channel,
    )

    assert link.link_id == "link1"
    assert link.source_node_id == "alice"
    assert link.target_node_id == "bob"
    assert link.port_kind is PortKind.CLASSICAL
    assert link.transport is channel


def test_network_link_rejects_invalid_port_kind() -> None:
    with pytest.raises(TypeError, match="port_kind must be PortKind"):
        NetworkLink(
            link_id="link1",
            source_node_id="alice",
            target_node_id="bob",
            port_kind="classical",  # type: ignore[arg-type]
        )
