"""Node-local namespaces for agents, devices, and component-owned ports.

Nodes provide stable names for control agents and devices, plus aliases for
existing component ports. They do not own runtime behavior, ports, or event
handling; those remain on the registered component objects.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from simyuj.components.ports import Port
from simyuj.primitives.ids import ensure_nonempty_id


class Node:
    """Topology namespace for node-local agents, devices, and ports.

    The node does not own runtime ports. Components own ports.
    The node only gives local names to existing component-owned ports.
    Control agents assigned to the node are activated by ``SessionRuntime``,
    not by the node itself.

    Parameters
    ----------
    node_id : str
        Stable node identifier used by network and routing helpers.

    Raises
    ------
    TypeError
        If ``node_id`` is not a string.
    ValueError
        If ``node_id`` is empty.
    """

    __slots__ = ("_node_id", "_agents", "_devices", "_ports")

    def __init__(self, node_id: str) -> None:
        self._node_id = ensure_nonempty_id(node_id, field_name="node_id")
        self._agents: dict[str, object] = {}
        self._devices: dict[str, object] = {}
        self._ports: dict[str, Port] = {}

    @property
    def node_id(self) -> str:
        """Return this node's stable identifier."""

        return self._node_id

    @property
    def agents(self) -> Mapping[str, object]:
        """Read-only mapping of agent ID to node-local control agent."""

        return MappingProxyType(self._agents)

    @property
    def devices(self) -> Mapping[str, object]:
        """Read-only mapping of node-local device name to component object."""

        return MappingProxyType(self._devices)

    @property
    def ports(self) -> Mapping[str, Port]:
        """Read-only mapping of node-local alias to component-owned port."""

        return MappingProxyType(self._ports)

    def add_device(self, name: str, device: object) -> object:
        """Register a named device or component under this node.

        Registered objects are not required to subclass ``Component``. If the
        node is added to a ``Network``, ``Network.bind_all`` will bind
        registered devices that expose a supported ``bind`` hook.

        Parameters
        ----------
        name : str
            Node-local device name.
        device : object
            Component or other runtime object to store.

        Returns
        -------
        object
            The same object supplied as ``device``.

        Raises
        ------
        ValueError
            If the device name is empty or already registered.
        """

        device_name = ensure_nonempty_id(name, field_name="device_name")

        if device_name in self._devices:
            raise ValueError(f"device name '{device_name}' already exists on node")

        self._devices[device_name] = device
        return device

    def add_agent(self, agent: object) -> object:
        """Register a control agent assigned to this node.

        ``SessionRuntime`` discovers registered agents and owns their lifecycle.
        Registering an agent on the node does not bind, start, or schedule it.

        Parameters
        ----------
        agent : Agent
            Control-plane agent assigned to this node. ``NodeAgent`` instances
            must have a ``node_id`` matching this node.

        Returns
        -------
        object
            The same object supplied as ``agent``.

        Raises
        ------
        TypeError
            If ``agent`` is not a control ``Agent``.
        ValueError
            If the agent ID is empty, duplicated on this node, or a
            ``NodeAgent`` is assigned to a different node.
        """

        from simyuj.control.agent import Agent, NodeAgent

        if not isinstance(agent, Agent):
            raise TypeError("agent must be control.Agent")

        agent_id = ensure_nonempty_id(agent.agent_id, field_name="agent_id")

        if isinstance(agent, NodeAgent) and agent.node_id != self._node_id:
            raise ValueError(
                f"agent node_id '{agent.node_id}' does not match node "
                f"'{self._node_id}'"
            )

        if agent_id in self._agents:
            raise ValueError(f"agent id '{agent_id}' already exists on node")

        self._agents[agent_id] = agent
        return agent

    def get_agent(self, agent_id: str) -> object:
        """Return a previously registered agent by agent ID.

        Raises
        ------
        KeyError
            If no agent is registered under ``agent_id``.
        """

        resolved_agent_id = ensure_nonempty_id(agent_id, field_name="agent_id")
        return self._agents[resolved_agent_id]

    def get_device(self, name: str) -> object:
        """Return a previously registered device by local name.

        Raises
        ------
        KeyError
            If no device is registered under ``name``.
        """

        device_name = ensure_nonempty_id(name, field_name="device_name")
        return self._devices[device_name]

    def register_port(self, name: str, port: Port) -> Port:
        """Register an alias to a component-owned port.

        Parameters
        ----------
        name : str
            Node-local alias for user code that wants to name a device port.
            Topology links do not depend on these aliases.
        port : Port
            Existing component-owned port. Ownership is not transferred to the
            node.

        Returns
        -------
        Port
            The same port supplied by the caller.

        Raises
        ------
        TypeError
            If ``port`` is not a ``Port``.
        ValueError
            If the alias is empty or already registered.
        """

        port_name = ensure_nonempty_id(name, field_name="port_name")

        if not isinstance(port, Port):
            raise TypeError("port must be components.Port")

        if port_name in self._ports:
            raise ValueError(f"port alias '{port_name}' already exists on node")

        self._ports[port_name] = port
        return port

    def get_port(self, name: str) -> Port:
        """Return a previously registered component-owned port alias.

        Raises
        ------
        KeyError
            If no port is registered under ``name``.
        """

        port_name = ensure_nonempty_id(name, field_name="port_name")
        return self._ports[port_name]


__all__ = ["Node"]
