"""Node-local device and port lookup for control-plane agents.

``DeviceResolver`` is attached to ``AgentContext`` for agents registered on a
``Node``. It exposes the aliases already registered on that node and does not
perform any scheduling or binding.
"""

from __future__ import annotations

from simyuj.components.memories import QuantumMemory
from simyuj.components.ports import Port
from simyuj.network import Node


class DeviceResolver:
    """Resolve devices and ports from one node namespace.

    Parameters
    ----------
    node : Node
        Network node whose device and port aliases are exposed.

    Notes
    -----
    Lookup errors are delegated to the underlying ``Node`` methods. The
    ``memory`` convenience method adds a type check for ``QuantumMemory``.
    """

    def __init__(self, *, node: Node) -> None:
        if not isinstance(node, Node):
            raise TypeError("node must be Node")
        self._node = node

    def get(self, name: str) -> object:
        """Return the node-local device registered as ``name``."""
        return self._node.get_device(name)

    def memory(self, name: str) -> QuantumMemory:
        """Return a node-local quantum memory device.

        Parameters
        ----------
        name : str
            Node-local device alias.

        Returns
        -------
        QuantumMemory
            Registered memory component.

        Raises
        ------
        TypeError
            If the alias resolves to a non-memory device.
        """
        device = self.get(name)
        if not isinstance(device, QuantumMemory):
            raise TypeError(f"device {name!r} is not QuantumMemory")
        return device

    def port(self, name: str) -> Port:
        """Return the node-local port registered as ``name``."""
        return self._node.get_port(name)


__all__ = ["DeviceResolver"]
