Ports And Connections
=====================

Ports and connections are the wiring layer for components.

A port belongs to one component and names a place where data can enter or
leave. A connection joins one output port to one input port. When a component
transmits through a connection, SimYuj schedules a timeline event for the
component on the other side.

Ports do not run code. They do not inspect payloads, apply delay, or call
``handle_event()``. They are structural endpoints. Components and the timeline
do the actual work.

The Mental Model
----------------

A simple component path looks like this:

.. code-block:: text

   source output -> channel input
   channel output -> detector input

Each arrow is a ``PortConnection``. Each connected port pair must agree on the
payload plane: quantum ports connect to quantum ports, and classical ports
connect to classical ports.

Ports
-----

A ``Port`` records four things:

- its local name, such as ``"in"``, ``"out"``, ``"left"``, or ``"report"``,
- the component that owns it,
- whether it is quantum or classical,
- whether it is an input or output endpoint.

Ports use object identity. A receiving component should check that a delivery
arrived at the exact port object it expects.

Connections
-----------

A ``PortConnection`` is one-way. It connects one output port to one input port.

Connection construction checks the important wiring mistakes early: wrong
direction, mismatched quantum/classical kind, same-component endpoints, and
ports that are already connected.

A port can have only one runtime connection at a time.

Delivery
--------

Calling ``transmit()`` on a connection does not immediately call the next
component. It schedules an ``Event`` on the timeline.

The scheduled event targets the owner of the input port. Its payload is a
``PortDelivery`` wrapper containing:

- the original payload,
- the source port,
- the target port,
- the connection ID.

Receiving components should use ``PortDelivery`` for correctness.
``Event.meta`` is useful for tracing and debugging, but it should not be the
source of truth for which port received the payload.

Developer Notes
---------------

Use ports when you are defining a reusable component interface. Use connections
when you are wiring components together for a simulation.

Keep component behavior inside the component. A connection should only
translate "a payload left this output port" into "an event will arrive at that
input port".

For bidirectional wiring, prefer paired one-way ``PortConnection`` objects, such
as ``a.out -> b.in`` and ``b.out -> a.in``. Avoid adding a bidirectional
``PortDirection`` unless the port connection ownership model changes to support
separate incoming and outgoing connection state.

Module Pages
------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Ports <ports>
   Connections <connections>
