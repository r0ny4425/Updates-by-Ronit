# Writing your own protocol

This page is not a full protocol design guide. It is a map for reading this
BB84 example and using the same shape for another protocol.

The main idea in SimYuj is simple:

```text
devices produce reports
agents keep protocol state
agents send public messages
agents schedule local events
the runtime executes everything on one timeline
```

If that structure is clear, the rest of the code is easier to follow.

## What belongs where

This BB84 example is split like this:

```text
configs.py      defaults and tunable parameters
helpers.py      small pure functions
agents.py       Alice and Bob state machines
trial.py        network/device/channel wiring
reporting.py    summary and JSON report helpers
demo.py         command-line entry point
```

That split is the part worth copying. The exact BB84 logic is protocol-specific.

## Agents are the protocol

The agents are the part that usually looks large. That is normal. They hold
the protocol state and decide what happens next.

A protocol agent usually has this shape:

```python
class MyProtocolAgent(NodeAgent):
    def on_start(self, start, ctx):
        ...

    def on_report(self, report, ctx):
        ...

    def on_message(self, message, ctx):
        ...

    def on_event(self, event, ctx):
        ...
```

Use the methods for different kinds of input:

```text
on_start    start a device, seed state, or schedule the first local action
on_report   react to source, detector, memory, or other device reports
on_message  react to public classical messages from another agent
on_event    react to local protocol events scheduled on the timeline
```

In this BB84 example:

- Alice starts the source in `on_start`.
- Alice records source reports in `on_report`.
- Bob records detector reports in `on_report`.
- Alice and Bob advance post-processing through `on_message`.
- Bob schedules local Cascade steps through `on_event`.

The notebook does not manually call `start_sifting()` or `start_privacy()`.
Those transitions happen because messages and local events arrive.

## Prefer visible protocol messages

If one party learns something, ask how they learned it.

In this example, Bob does not inspect Alice's source object to know that the
quantum frame is over. Alice sends:

```text
quantum.done
```

Bob also does not sift using Alice's private signal ids. He announces detected
slot indices and bases:

```text
sift.bob_bases
```

Alice replies with accepted slots:

```text
sift.accepted
```

This makes the event log useful. A reader can see the public conversation that
drives the protocol.

## Keep `trial.py` explicit

`trial.py` is allowed to be a little long because it answers an important
question: what is connected to what?

For a new protocol, this file should usually create:

```text
devices
agents
nodes
quantum links
classical links
port wires
runtime
logger
```

The wiring should be plain and explicit. Hidden wiring saves a few lines but
makes examples harder to trust.

## Start smaller than the final protocol

When building a new protocol, do not begin with everything at once. A useful
order is:

1. Make the devices run and produce reports.
2. Add one agent and record those reports.
3. Add the second agent.
4. Add one public message.
5. Add one local event.
6. Add the first real protocol stage.
7. Only then add the later stages.

For BB84, that means source and detector first, then sifting, then QBER, then
reconciliation, then verification, then privacy amplification.

Small runs are useful even when they abort. For example, a 20-slot run may not
have enough sifted bits for QBER estimation. That is not a failure of the
simulator if the report says clearly:

```text
alice_abort: not enough sifted bits ...
bob_abort: not enough sifted bits ...
```

## What to check while debugging

For a physical protocol, counters are your friend. In this example, check:

```text
prepared_photons
no_emission_slots
channel_delivered
channel_lost
bob_detections
bob_failed_reports
bob_unassigned_reports
sifted_bits
estimated_qber
final_keys_equal
alice_abort
bob_abort
```

Also save a JSONL event log when the run is confusing. It will show the order
of device reports, messages, and local events.

## Adapt the pattern
This folder is a BB84 single-photon example. Another protocol may need
different pieces: an entanglement source, memories, more nodes, a different
receiver, decoy states, or a different security analysis.

The reusable pattern is:

```text
make assumptions visible
wire the network explicitly
let agents advance by events and messages
return a report that explains what happened
```

That is the part to carry into the next protocol.
