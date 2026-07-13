Components
==========

Components are the device-like objects in a SimYuj simulation. They represent
the physical pieces that can emit, carry, store, transform, or measure signals:
sources, channels, detectors, memories, and small quantum targets.

In a network simulation, components usually live inside nodes or inside the
connections between nodes. A source may sit at one node, a channel may belong to
a link, and a detector or memory may receive the signal at the other end.

Most reusable devices build on the engine's ``Component`` model. Ports describe
where the component can send or receive data, while connections and the timeline
decide when another component actually sees that data.

A useful way to read this package is:

.. code-block:: text

   source -> channel -> detector or memory

What Belongs Here
-----------------

Use components for reusable simulator behavior:

- sources that create quantum signals,
- channels that delay, lose, or transform signals,
- detectors that turn arrivals into measurement reports,
- memories that absorb, store, retrieve, or expire quantum state.

Protocol decisions should usually live above this layer. A protocol may choose
when to fire a source or read a detector result, but the component should own
the physical or device-like behavior itself.

How Components Interact
-----------------------

Components communicate through ports and scheduled events. A component should
not directly call another component's event handler. It should transmit through
a connection, or schedule work on the timeline.

This keeps simulations replayable: the same seed, configuration, and event
order should give the same result.

Common Starting Points
----------------------

Start with ports and connections if you are building a new device. They explain
how components are wired together.

Read sources, channels, detectors, and memories when you want to model a
specific physical part of a simulation.

Read quantum targets when you need a small object that can receive quantum
signals without becoming a full device model.

Developer Notes
---------------

Keep components focused. A good component has a clear device-like role and a
small event surface. If it starts making protocol choices, routing decisions,
or post-processing decisions, that logic probably belongs in another package.

Module Pages
------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   Ports and Connections <ports_connections>
   Quantum Targets <quantum_targets>
   Channels <channels>
   Sources <sources>
   Detectors <detectors>
   Memories <memories>
