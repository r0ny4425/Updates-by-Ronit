Engine
======

The engine package is SimYuj's deterministic discrete-event kernel. It owns
simulation time, event ordering, event identifiers, batch execution, lazy
cancellation, deterministic random streams, timeline-owned qstate access,
execution statistics, and observational logging.

The engine is deliberately small. It does not know what a protocol, device,
network link, memory, or detector means. Those meanings live in higher layers.
The engine's job is to answer one question reliably:

When several things are scheduled to happen, what happens next?

Overview
--------

Most simulations use the engine through two objects:

``Timeline``
   The execution authority. It assigns event ids, advances simulation time,
   extracts same-time batches, dispatches events to targets, owns deterministic
   RNG streams, and records execution summaries.

``Event``
   A scheduled causal record. It says that at integer simulation time
   ``time``, the timeline should deliver ``action`` and ``payload_ref`` to
   ``target_ref``.

An event target is any object with ``handle_event(event, timeline)``. Targets
interpret actions and payloads for their own domain. When they need something
else to happen, they schedule another event through the timeline.

A Small Example
---------------

This example models a coffee shop. A cashier receives an order, schedules a
barista to prepare it, and the barista schedules an audit-log event when the
order is ready.

.. code-block:: pycon

   >>> from simyuj.engine import Event, Timeline


   >>> class AuditLog:
   ...     def __init__(self):
   ...         self.records = []
   ...
   ...     def handle_event(self, event, timeline):
   ...         self.records.append(
   ...             (timeline.current_time, event.action, event.payload_ref)
   ...         )

   >>> class Cashier:
   ...     def __init__(self, barista):
   ...         self.barista = barista
   ...
   ...     def handle_event(self, event, timeline):
   ...         order = event.payload_ref
   ...
   ...         timeline.schedule(
   ...             Event(
   ...                 time=timeline.current_time + 3,
   ...                 target_ref=self.barista,
   ...                 action="prepare_order",
   ...                 payload_ref=order,
   ...                 source=self,
   ...             )
   ...         )

   >>> class Barista:
   ...     def __init__(self, audit):
   ...         self.audit = audit
   ...
   ...     def handle_event(self, event, timeline):
   ...         order = event.payload_ref
   ...
   ...         timeline.schedule(
   ...             Event(
   ...                 time=timeline.current_time,
   ...                 priority=-10,
   ...                 target_ref=self.audit,
   ...                 action="order_ready",
   ...                 payload_ref=order,
   ...                 source=self,
   ...             )
   ...         )


   >>> timeline = Timeline()

   >>> audit = AuditLog()
   >>> barista = Barista(audit)
   >>> cashier = Cashier(barista)

   >>> _ = timeline.schedule(
   ...     Event(
   ...         time=0,
   ...         target_ref=cashier,
   ...         action="take_order",
   ...         payload_ref="latte",
   ...     )
   ... )

   >>> summaries = timeline.run_until_empty()

   >>> audit.records
   [(3, 'order_ready', 'latte')]
   >>> [summary.batch_time for summary in summaries]
   [0, 3, 3]

The cashier does not call the barista directly. It schedules an event. The
barista also schedules the audit-log event, even though it happens at the same
simulation time.

Because ``order_ready`` is created while the ``prepare_order`` batch is already
running, it executes in a later batch at the same timestamp. This keeps batch
execution deterministic: handlers can schedule follow-up work, but they cannot
change the batch they are currently inside.

Execution Lifecycle
-------------------

A typical engine workflow is:

1. Create a ``Timeline`` with an optional ``master_seed``.
2. Declare any deterministic RNG streams with ``timeline.rng(...)`` before
   execution starts.
3. Schedule one or more ``Event`` objects with ``timeline.schedule(...)``.
4. Execute one batch with ``run_one_step()``, run to a time boundary with
   ``run_until(t_end)``, or run until no active events remain with
   ``run_until_empty()``.
5. Inspect returned ``ExecutionSummary`` records or the immutable
   ``timeline.stats`` snapshot.

The first execution step freezes RNG stream creation. Existing streams remain
usable, but new stream names cannot be introduced after execution begins.

Ordering and Batches
--------------------

The queue orders events by:

1. ``time``
2. ``priority``
3. timeline-assigned ``event_id``

Lower values execute first. ``event_id`` is the final tie-breaker, so events
with the same time and priority execute in scheduling order.

``run_one_step()`` executes exactly one batch: all active events that share the
earliest executable timestamp when the step begins. Events scheduled by a
handler are not inserted into the batch currently executing. If they are
scheduled for the current timestamp, they become a later batch at the same
time.

Handler errors propagate to the caller. Batch execution is not rolled back:
events already dispatched stay dispatched, and events already removed from the
batch are not automatically restored.

``run_until(t_end)`` executes every whole batch whose timestamp is less than or
equal to ``t_end``. The first batch after ``t_end`` remains queued.

Cancellation and Rescheduling
-----------------------------

``Timeline.cancel(event)`` marks an event as cancelled. Cancellation is lazy:
the event may remain in the raw heap until it reaches the front, but it will
not execute.

``Timeline.reschedule(...)`` cancels the original event and schedules a
replacement with a fresh timeline-assigned ``event_id``.

Randomness
----------

Stochastic behavior should use named streams from ``Timeline.rng(...)``:

.. code-block:: pycon

   >>> from simyuj.engine import Timeline
   >>> timeline = Timeline(master_seed=123)
   >>> arrivals = timeline.rng("coffee_shop", "arrivals")
   >>> service = timeline.rng("coffee_shop", "service_time")
   >>> arrivals is service
   False

The same master seed and stream path produce the same sequence across runs.
Stream creation order does not affect the sequence. Stream names are
hierarchical, case-sensitive, and must be declared before execution begins.

Component Rules
---------------

Event targets should:

* implement ``handle_event(event, timeline)``;
* treat events as read-only causal records;
* schedule follow-up work through ``timeline.schedule(...)``;
* route stochastic choices through declared ``timeline.rng(...)`` streams;
* let the timeline own time advancement and event ordering.

Event targets should not:

* mutate ``timeline.current_time``;
* call another component's ``handle_event`` directly;
* create new RNG stream paths after execution starts;
* hide cross-component behavior outside scheduled events.

Observability
-------------

``run_one_step()`` returns an ``ExecutionSummary`` for the executed batch:
batch time, number of dispatched events, and event ids in dispatch order.

``timeline.stats`` returns an immutable ``TimelineStatistics`` snapshot:
scheduled count, executed count, maximum raw queue size, and current time.
Reading statistics is observational and does not affect execution.

Boundary Rules
--------------

The engine owns scheduling mechanics. It does not own domain meaning.

Use the engine for:

* deterministic event ordering;
* time advancement;
* batch execution;
* lazy cancellation;
* named deterministic RNG streams;
* execution summaries, statistics, and timeline-level logging.

Keep outside the engine:

* protocol decisions;
* device physics;
* network routing policy;
* quantum-state math;
* post-processing workflows;
* application-specific action names and payload semantics.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Component <component>
   Event <event>
   Event Ordering <event_ordering>
   Event Queue <event_queue>
   Execution Summary <execution_summary>
   RNG Manager <rng_manager>
   Timeline <timeline>
   Timeline Statistics <timeline_statistics>
