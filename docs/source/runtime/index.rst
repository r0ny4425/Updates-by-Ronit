Runtime
=======

The runtime package provides SimYuj's deterministic setup phase.

Use it when an object needs to declare run-scoped resources before the
timeline starts executing events. The most common example is requesting named
RNG streams during binding, before ``Timeline`` freezes stream creation.

Runtime setup is deliberately separate from event execution. Binding prepares
objects for a run; it does not advance time, deliver messages, or call another
component's ``handle_event()`` method.

When To Use Binding
-------------------

Use a ``bind(context)`` hook when an object needs access to run-scoped
services:

* the active ``Timeline``;
* the simulation logger;
* deterministic RNG streams;
* session, component, or caller metadata.

Do not use binding for protocol behavior. Protocol behavior should still
happen through scheduled events.

A Small Example
---------------

This object needs a deterministic random stream, so it declares the stream
during binding instead of creating it inside ``handle_event()``:

.. code-block:: pycon

   >>> from simyuj.engine import Timeline
   >>> from simyuj.runtime import BindingContext, bind_many


   >>> class JitterModel:
   ...     def bind(self, context: BindingContext) -> None:
   ...         self.rng = context.timeline.rng("example", "jitter")
   ...
   ...     def sample_delay(self) -> int:
   ...         return self.rng.randint(1, 3)

   >>> timeline = Timeline(master_seed=7)
   >>> jitter = JitterModel()

   >>> bound = bind_many([jitter], timeline)

   >>> bound == (jitter,)
   True
   >>> jitter.sample_delay()
   2

Binding runs before the first execution step. After execution starts, new RNG
stream names cannot be introduced.

Binding Context
---------------

``BindingContext`` is an immutable record passed to optional ``bind(context)``
hooks. It carries:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``timeline``
     - Timeline whose deterministic runtime resources should be declared.
   * - ``logger``
     - Simulation logger supplied by the caller or resolved from the timeline.
   * - ``entity_id``
     - Optional higher-level entity identifier, such as a session ID.
   * - ``component_id``
     - Optional component, device, or role identifier.
   * - ``meta``
     - Immutable caller-supplied metadata entries.

The context intentionally does not validate concrete timeline or logger types
at construction time. Callers and bind hooks are responsible for passing
compatible objects.

Lifecycle
---------

A typical setup flow is:

1. Create the ``Timeline``.
2. Construct components, channels, agents, or runtime objects.
3. Bind objects that expose ``bind(context)`` before execution starts.
4. Schedule initial events.
5. Run the timeline.

Binding is not a replacement for events. If setup discovers work that should
happen during the simulation, schedule that work as an event.

Helpers
-------

``bind_if_supported`` inspects one object. If the object has no ``bind``
attribute, it returns ``False``. If ``bind`` exists but is not callable, it
raises ``TypeError``. Otherwise it builds a ``BindingContext``, calls the bind
method, and returns ``True``.

``bind_many`` applies that logic across an iterable in iteration order. It
returns only the entities whose bind hooks were called and skips objects that
do not support binding.

``BindableMixin`` is an optional no-op base for classes that want a stable
``bind`` slot. ``SupportsBind`` is the runtime-checkable protocol for objects
that expose ``bind(context)``.

Where It Appears
----------------

Network and session runtimes build on this primitive. For example,
``Network.bind_all`` visits registered devices in deterministic node/device
order, and ``SessionRuntime.bind_all`` binds network devices before
node-registered agents.

Boundary Rules
--------------

The runtime package is lifecycle glue. It passes setup context to objects, and
those objects decide what they declare and what events they later schedule.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Binding <binding>
