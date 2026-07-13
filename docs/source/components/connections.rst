Connections
===========

Connections turn port-level transmission into timeline events. A connection
links one egress port to one ingress port and schedules a ``PortDelivery`` event
targeted at the receiving component.

Ports identify the endpoints being crossed. Connections own the runtime routing
between those endpoints. The target of a scheduled event is always the owner of
the target port, never the port object itself.

Basic Flow
----------

A connected transmission follows this shape:

.. code-block:: text

   source component
      -> egress Port
      -> PortConnection.transmit(...)
      -> Timeline event carrying PortDelivery
      -> target component receives event through ingress Port

The original payload is wrapped in ``PortDelivery`` together with the source
port, target port, and connection id. Receiving components should inspect that
wrapper when they need to validate which local port received the payload.

What Connections Validate
-------------------------

``connect_ports(...)`` and ``PortConnection`` validate the wiring before a
runtime connection is installed:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Rule
     - Reason
   * - Source must be an ``EGRESS`` port.
     - Connections are one-way from sender to receiver.
   * - Target must be an ``INGRESS`` port.
     - Scheduled events are delivered to the target port owner.
   * - Port kinds must match.
     - Classical and quantum payload planes are not mixed by the connection
       layer.
   * - Source and target owners must differ.
     - Connections model component-to-component routing.
   * - Each endpoint may have only one connection.
     - Wiring is one-to-one and explicit.

On success, the connection stores itself on both endpoint ports through their
``connection`` attributes.

Scheduling Semantics
--------------------

``PortConnection.transmit(...)`` schedules an ``Event`` on the provided
``Timeline``. The ``time`` argument is an absolute simulation tick. If omitted,
the delivery is scheduled at the current timeline time.

The event action is normally the connection's ``target_action``. Callers may
override it for a specific transmission by passing ``action=...``.

Connections do not call ``handle_event(...)`` directly. Delivery remains visible
to the deterministic event engine, so ordering, priority, event ids, and replay
stay under ``Timeline`` control.

Payload And Metadata
--------------------

The scheduled event payload is a fresh ``PortDelivery``:

.. code-block:: python

   PortDelivery(
       payload=payload,
       source_port=source_port,
       target_port=target_port,
       connection_id=connection_id,
   )

``Event.meta`` also receives connection tracing fields such as
``connection_id``, ``source_port``, ``target_port``, and ``target_action``.
Caller-supplied metadata is merged after those defaults and may override them.

Use ``PortDelivery`` for correctness-sensitive routing checks. Treat
``Event.meta`` as trace/debug context.

Connection Identifiers
----------------------

When no explicit ``connection_id`` is supplied, ``connect_ports(...)`` derives
one from the endpoint owner ids and port names:

.. code-block:: text

   source_owner.source_port->target_owner.target_port

This id is stored on ``PortDelivery`` and copied into event metadata.

Missing Connections
-------------------

Components commonly call ``require_connection(port)`` immediately before
transmitting through a required output port. If the port is unconnected, the
helper raises ``RuntimeError`` with the fully qualified port name.

Use this when an unconnected port is a runtime configuration error rather than
an optional output path.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/connections.py``

.. automodule:: simyuj.components.connections
   :members:
   :show-inheritance:
