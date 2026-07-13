from __future__ import annotations

from simyuj.engine.timeline import Timeline
from simyuj.network import Network, Node
from simyuj.runtime.binding import BindingContext
from tests.support.binding import binding_context


class BindRecorder:
    def __init__(self) -> None:
        self.bound_contexts: list[BindingContext] = []

    def bind(self, context: BindingContext) -> None:
        self.bound_contexts.append(context)


class NonBindable:
    pass


def test_network_bind_all_binds_registered_devices_once() -> None:
    timeline = Timeline(master_seed=1)

    network = Network("bind_graph")

    alice = Node("alice")
    bob = Node("bob")

    shared = BindRecorder()
    other = BindRecorder()
    non_bindable = NonBindable()

    alice.add_device("shared", shared)
    alice.add_device("non_bindable", non_bindable)
    bob.add_device("shared_again", shared)
    bob.add_device("other", other)

    network.add_node(alice)
    network.add_node(bob)

    bound = network.bind_all(binding_context(timeline))

    assert [context.timeline for context in shared.bound_contexts] == [timeline]
    assert [context.timeline for context in other.bound_contexts] == [timeline]
    assert dict(shared.bound_contexts[0].meta) == {
        "node_id": "alice",
        "device_id": "shared",
    }
    assert dict(other.bound_contexts[0].meta) == {
        "node_id": "bob",
        "device_id": "other",
    }
    assert non_bindable not in bound
    assert set(bound) == {shared, other}


def test_network_bind_all_binds_link_transports_once() -> None:
    timeline = Timeline(master_seed=1)

    network = Network("bind_links")
    network.add_node(Node("alice"))
    network.add_node(Node("bob"))
    network.add_node(Node("carol"))

    shared_transport = BindRecorder()
    other_transport = BindRecorder()

    network.add_quantum_link(
        "q_ab",
        "alice",
        "bob",
        channel=shared_transport,
    )
    network.add_quantum_link(
        "q_bc",
        "bob",
        "carol",
        channel=shared_transport,
    )
    network.add_classical_link(
        "c_ac",
        "alice",
        "carol",
        channel=other_transport,
    )

    bound = network.bind_all(binding_context(timeline))

    assert [context.timeline for context in shared_transport.bound_contexts] == [
        timeline
    ]
    assert [context.timeline for context in other_transport.bound_contexts] == [
        timeline
    ]
    assert dict(shared_transport.bound_contexts[0].meta) == {
        "link_id": "q_ab",
    }
    assert dict(other_transport.bound_contexts[0].meta) == {
        "link_id": "c_ac",
    }
    assert set(bound) == {shared_transport, other_transport}
