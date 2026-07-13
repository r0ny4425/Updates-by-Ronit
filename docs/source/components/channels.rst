Channels
========

Channel components move payloads between components. They are the transport
devices in a simulation: a classical channel carries classical messages, and a
quantum channel carries qstate-backed signals.

A channel has an input port and an output port. The input side receives a
payload from an upstream connection. The output side schedules delivery to the
next component through another connection.

Channels do not call downstream components directly. Delivery is represented as
a timeline event, so propagation delay, event ordering, and replay stay visible.

What Channels Model
-------------------

Use channels when the path between two components should affect the payload:

- propagation delay,
- message or signal loss,
- quantum attenuation,
- timing jitter,
- qstate noise over the channel duration.

``ClassicalChannel`` is a compact delay-and-loss model for classical messages.
It forwards the original message object when the message survives.

``QuantumChannel`` is a compact signal-level fiber model. It can drop a signal,
apply configured qstate noise to surviving targets, and append channel timing
metadata before forwarding the signal.

Classical channels leave qstate untouched. Quantum channels work through the
explicit targets carried by the signal. If a signal is lost, those targets are
discarded through the timeline qstate manager; if it survives, the same
subsystem identity is forwarded after any configured channel noise is applied.

Basic Flow
----------

A channel usually behaves like this:

.. code-block:: text

   receive payload -> sample loss -> apply channel effects -> schedule delivery

The channel itself is the event target. Ports are wiring points; they do not
handle events.

Timing
------

Channels can use an explicit ``delay_ticks`` value, or derive delay from fiber
length and propagation speed. Under the default unit scale, one simulation tick
is one picosecond.

The current channel models represent propagation by the scheduled time of the
downstream event. They do not create a separate internal "in flight" event.

Use the detailed module pages when you need constructor arguments, event action
names, metadata fields, or the exact loss and timing assumptions.

Module Pages
------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Classical Channel <channels/classical>
   Quantum Channel <channels/quantum>
   Shared Channel Helpers <channels/common>
