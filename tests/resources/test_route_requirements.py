import pytest

from simyuj.components.ports import PortKind
from simyuj.network.routing import Route
from simyuj.network.topology import TopologyEdge
from simyuj.resources.manager import ResourceManager
from simyuj.resources.route_requirements import (
    NodeMemoryRequirement,
    requirements_mapping,
    reserve_route_memories,
    route_memory_requirements,
)


def route(*node_ids: str) -> Route:
    if len(node_ids) < 1:
        raise ValueError("route needs at least one node")

    edges = []
    for i in range(len(node_ids) - 1):
        edges.append(
            TopologyEdge(
                link_id=f"l_{node_ids[i]}_{node_ids[i+1]}",
                source_node_id=node_ids[i],
                target_node_id=node_ids[i + 1],
                port_kind=PortKind.QUANTUM,
            )
        )

    return Route(node_ids[0], node_ids[-1], tuple(edges))


def test_node_memory_requirement_validates_fields() -> None:
    requirement = NodeMemoryRequirement("alice", 2)

    assert requirement.node_id == "alice"
    assert requirement.requirement == 2


def test_route_memory_requirements_builds_sorted_counts() -> None:
    requirements = route_memory_requirements(
        route("alice", "r2", "r1", "bob"),
        node_requirements=lambda node, idx, length: 1 if idx in (0, length - 1) else 2,
    )

    assert requirements == (
        NodeMemoryRequirement("alice", 1),
        NodeMemoryRequirement("bob", 1),
        NodeMemoryRequirement("r1", 2),
        NodeMemoryRequirement("r2", 2),
    )


def test_route_memory_requirements_omits_zero_counts() -> None:
    assert route_memory_requirements(
        route("alice", "relay", "bob"),
        node_requirements=lambda node, idx, length: 1 if idx in (0, length - 1) else 0,
    ) == (
        NodeMemoryRequirement("alice", 1),
        NodeMemoryRequirement("bob", 1),
    )


def test_route_memory_requirements_merges_repeated_device_mappings() -> None:
    requirements = route_memory_requirements(
        route("alice", "relay", "alice"),
        node_requirements=lambda node, idx, length: (
            {"qmem_a": 1} if node == "alice" else {}
        ),
    )

    assert requirements == (NodeMemoryRequirement("alice", {"qmem_a": 2}),)


def test_route_memory_requirements_rejects_mixed_repeated_requirements() -> None:
    with pytest.raises(ValueError, match="cannot merge integer requirement"):
        route_memory_requirements(
            route("alice", "relay", "alice"),
            node_requirements=lambda node, idx, length: (
                1 if idx == 0 else {"qmem_a": 1}
            ),
        )


def test_route_memory_requirements_validates_inputs() -> None:
    with pytest.raises(TypeError, match="route must be Route"):
        route_memory_requirements(
            "not-route",  # type: ignore[arg-type]
            node_requirements=lambda node, index, link: 1,
        )


def test_requirements_mapping_extracts_counts() -> None:
    requirements = (
        NodeMemoryRequirement("alice", 1),
        NodeMemoryRequirement("bob", {"qmem_a": 2}),
    )

    mapping = requirements_mapping(requirements)

    assert mapping == {
        "alice": 1,
        "bob": {"qmem_a": 2},
    }


def test_reserve_route_memories_reserves_selected_node_slots() -> None:
    manager = ResourceManager()
    alice_refs = manager.register_memory("alice", "qmem", num_positions=1)
    relay_refs = manager.register_memory("relay", "qmem", num_positions=2)
    bob_refs = manager.register_memory("bob", "qmem", num_positions=1)

    reservation = reserve_route_memories(
        10,
        manager,
        route("alice", "relay", "bob"),
        node_requirements=lambda node, idx, length: 1 if idx in (0, length - 1) else 2,
        owner="session",
        reservation_id="reservation:route",
        metadata=(("kind", "route"),),
    )

    assert reservation.owner == "session"
    assert reservation.reservation_id == "reservation:route"
    assert reservation.metadata == (("kind", "route"),)
    assert reservation.memory_refs == (*alice_refs, *bob_refs, *relay_refs)


def test_reserve_route_memories_reserves_targeted_device_slots() -> None:
    manager = ResourceManager()
    alice_refs = manager.register_memory("alice", "qmem", num_positions=1)
    relay_a_refs = manager.register_memory("relay", "qmem_a", num_positions=2)
    relay_b_refs = manager.register_memory("relay", "qmem_b", num_positions=1)
    bob_refs = manager.register_memory("bob", "qmem", num_positions=1)

    reservation = reserve_route_memories(
        10,
        manager,
        route("alice", "relay", "bob"),
        node_requirements=lambda node, idx, length: (
            {"qmem_a": 2, "qmem_b": 1} if node == "relay" else 1
        ),
        owner="session",
    )

    assert reservation.memory_refs == (
        *alice_refs,
        *bob_refs,
        *relay_a_refs,
        *relay_b_refs,
    )


def test_route_link_ids_can_drive_actual_device_reservation() -> None:
    manager = ResourceManager()
    alice_link = manager.register_memory("alice", "mem_l_alice_relay", num_positions=1)
    relay_left = manager.register_memory("relay", "mem_l_alice_relay", num_positions=1)
    relay_right = manager.register_memory("relay", "mem_l_relay_bob", num_positions=1)
    bob_link = manager.register_memory("bob", "mem_l_relay_bob", num_positions=1)
    path = route("alice", "relay", "bob")

    def link_scoped_requirement(node: str, idx: int, length: int):
        del node
        if idx == 0:
            return {f"mem_{path.link_ids[0]}": 1}
        if idx == length - 1:
            return {f"mem_{path.link_ids[-1]}": 1}
        return {
            f"mem_{path.link_ids[idx - 1]}": 1,
            f"mem_{path.link_ids[idx]}": 1,
        }

    reservation = reserve_route_memories(
        10,
        manager,
        path,
        node_requirements=link_scoped_requirement,
        owner="session",
        reservation_id="reservation:link-scoped-route",
    )

    assert reservation.memory_refs == (
        *alice_link,
        *bob_link,
        *relay_left,
        *relay_right,
    )


def test_reserve_route_memories_rejects_insufficient_targeted_device_slots() -> None:
    manager = ResourceManager()
    manager.register_memory("alice", "qmem", num_positions=1)
    manager.register_memory("relay", "qmem_a", num_positions=1)
    manager.register_memory("bob", "qmem", num_positions=1)

    with pytest.raises(ValueError, match="device 'qmem_a'"):
        reserve_route_memories(
            10,
            manager,
            route("alice", "relay", "bob"),
            node_requirements=lambda node, idx, length: (
                {"qmem_a": 2} if node == "relay" else 1
            ),
            owner="session",
        )


def test_reserve_route_memories_rejects_empty_requirements() -> None:
    manager = ResourceManager()

    with pytest.raises(ValueError, match="requirements are empty"):
        reserve_route_memories(
            10,
            manager,
            route("alice"),
            node_requirements=lambda node, index, link: 0,
            owner="test-owner",
        )


def test_reserve_route_memories_validates_manager() -> None:
    with pytest.raises(TypeError, match="manager must be ResourceManager"):
        reserve_route_memories(
            10,
            object(),  # type: ignore[arg-type]
            route("alice", "bob"),
            node_requirements=lambda node, index, link: 1,
            owner="test-owner",
        )
