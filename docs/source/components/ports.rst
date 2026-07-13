Ports
=====

Ports are structural endpoints owned by components. They identify where
payloads enter or leave a component, but they do not handle events, inspect
payloads, or schedule timeline work.

Runtime delivery is handled by connections. A ``PortConnection`` targets the
owner of the target port and carries the original endpoint identities in a
``PortDelivery`` payload.

Port Roles
----------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Type
     - Meaning
   * - ``PortKind.QUANTUM``
     - Carries qstate-backed quantum signals.
   * - ``PortKind.CLASSICAL``
     - Carries classical messages, reports, or control payloads.
   * - ``PortDirection.INGRESS``
     - Input endpoint owned by the receiving component.
   * - ``PortDirection.EGRESS``
     - Output endpoint used by the sending component.

Runtime State
-------------

``Port.connection`` starts as ``None``. When ``PortConnection`` is created, it
installs itself on both endpoint ports. Components can then check
``port.is_connected`` before transmitting through optional outputs.

Ports use object identity semantics. This lets receiving components validate
the exact local endpoint that received a ``PortDelivery``.

``Port`` construction checks the local structural fields: non-empty name,
non-empty owner id, component owner, valid kind, and valid direction.

Direction matching, kind matching, one-to-one wiring, and distinct owners are
validated by the connection layer.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/ports.py``

.. automodule:: simyuj.components.ports
   :members:
   :show-inheritance:
