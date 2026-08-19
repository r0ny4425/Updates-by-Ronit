# simyuj — working notes for AI assistants

Deterministic discrete-event quantum network simulator. ~20k LOC.

## Read this first

A knowledge graph exists at `graphify-out/graph.json`. For any "how does X work",
"where does Y live", or "what connects A to B" question:

```
graphify query "<question>"
graphify explain "<NodeName>"
graphify path "<A>" "<B>"
```

Do **not** grep source for architecture questions. Read source only to confirm a
signature you are about to use. The graph has isolated nodes and low community
cohesion, so falling back to source is expected — leading with source is not.

For mapping a simulation requirement onto simyuj modules, read `CAPABILITY_MAP.md`
before exploring anything.

## The architectural rule

```
devices produce reports
agents keep protocol state
agents send public messages
agents schedule local events
the runtime executes everything on one timeline
```

Layer ownership:

| Layer | Package | Modify for a new protocol? |
|---|---|---|
| Engine | `engine/` | Never |
| Physics | `qstate/` | Rarely |
| Devices | `components/` | Only for genuinely new hardware |
| Topology | `network/`, `metrics/`, `entanglement/`, `resources/` | Configure, don't modify |
| **Protocol** | **`control/` — `NodeAgent` subclasses** | **Yes — work lives here** |

## Invariants — violating any of these is a bug, not a style choice

1. **Determinism.** All randomness comes from
   `Timeline.rng(device_id, component_key, stream_name)` — all three arguments,
   so two instances of the same component type cannot share a stream. Never
   `numpy.random`, never `random`, never an unseeded generator. Runs must replay
   identically from `master_seed`.
   Three segments is the floor, not a fixed arity: a device holding several
   identical sub-devices appends one more segment to keep them apart — see
   `components/detectors/primitives/window.py`, which uses
   `(device_id, namespace, detector_id, stream_name)`.
2. **Event discipline.** Components never call each other's methods directly. Every
   cross-component interaction is a scheduled timeline event — including `delay=0`
   timers and same-tick memory requests.
3. **No hidden ordering.** Nothing depends on dict/set iteration order, `id()`, or
   wall-clock time. Agent starts are ordered by sorted `agent_id`.
4. **Time belongs to the timeline.** Nothing advances time except the timeline.
5. **Layer boundaries.** `qstate` never schedules events or advances time. Components
   never contain protocol policy. Agents never do channel physics by hand.
6. **Protocol knowledge is earned.** If an agent knows something, a message told it.
   Never read another agent's private state.
7. **Logging is observational.** Tracing must never change the run.
8. **Physical units in configs**, converted with `seconds_to_ticks` at construction.

## Where to look for a pattern

| Task | Reference |
|---|---|
| Prepare-and-measure QKD | `examples/bb84/` (canonical, most complete) |
| Classical post-processing | `examples/postprocessing/bb84_event/` |
| Entanglement-based QKD | `tutorials/protocols/e91.ipynb` |
| Bell measurement + corrections | `tutorials/protocols/teleportation.ipynb` |
| Purification | `tutorials/protocols/distillation.ipynb` |
| Multi-node / concurrent | `tutorials/four_node_qkd/` |
| Any subsystem in isolation | `tutorials/01_engine` … `10_control` labs |

The method for writing a new protocol is documented in
`examples/bb84/docs/03_write_your_own_protocol.md`. Follow it rather than inventing
an approach.

## New protocol file layout

```
configs.py      frozen dataclasses, all tunable parameters, physical units
helpers.py      small pure functions
agents.py       NodeAgent subclasses — the protocol itself
trial.py        explicit wiring: devices, agents, nodes, links, wires, runtime, logger
reporting.py    summary and JSON report
demo.py         CLI entry point
```

Build in this order, one stage at a time, checking counters after each:

1. Devices run and produce reports
2. One agent records them
3. Second agent
4. One public message
5. One local scheduled event
6. First protocol stage
7. Remaining stages

A run that aborts with a clear reason is a success, not a failure.

## Testing

```bash
uv run pytest tests/ -q
uv run pytest tests/qstate -q -k "bell"
```

Mirror the existing suite structure. Use `tests/support/mock_components/` and
`tests/conftest.py` fixtures rather than inventing scaffolding. Assert deterministic
outcomes at a fixed `master_seed`. Sweep ≥20 seeds for any statistical claim.

**The stubs in `tests/support/mock_components/` are pre-port-layer.** `EmitterStub`,
`ChannelStub`, and `DetectorStub` take a bare `output_target: Component`, build
`Event(target_ref=...)` by hand, and validate `event.action` against a hardcoded
frozenset (`"source.emit.start"`, `"quantum.signal.out"`, `"quantum.signal.in"`).
They exist to exercise `Timeline` batching and dispatch, and for that they are still
correct — keep using them for engine-level tests.

They cannot sit on the receiving end of a `PortConnection`, for two reasons, and the
action strings are *not* one of them (`PortConnection.target_action` is arbitrary):

- They own no `Port`. `PortConnection` requires a `target_port: Port` whose `owner`
  is the component, so there is nothing to wire to.
- `PortConnection.transmit()` always wraps the payload in a `PortDelivery`. The stubs
  read `event.payload_ref` as the payload itself — `ChannelStub` would forward the
  wrapper downstream instead of the contents, and neither stub checks
  `delivery.target_port` against its own endpoint.

So a port-based test needs a **port-based sink added to `tests/support/mock_components/`**,
not a fresh one inlined per test file. That inlining has already happened: `QuantumSink` is
defined five separate times — `tests/components/memories/test_quantum_memory_e2e.py`,
`tests/components/memories/_quantum_memory_support.py`,
`tests/components/sources/test_entangled_pair_source.py`,
`tests/components/test_components_quantum_channel.py`, `tests/network/_components.py` — and
`ReportSink` four times: three under `tests/components/detectors/` plus
`tests/engine/test_end_to_end_topology.py`. Extend the shared folder rather than adding
another copy.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), imperative mood:

```
<type>(<optional scope>): <subject>
```

Types in use: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.
Subject in imperative mood ("add", not "added"/"adds"), no trailing period,
lower case after the colon. Body explains *why* and notes anything a reviewer
would otherwise have to reverse-engineer.

- One logical change per commit. Don't mix a rename with a behaviour change.
- Never commit to the default branch — branch first.
- Multi-phase work commits at each phase boundary, so a bad phase can be rolled
  back without losing the good ones.
- State physics or determinism implications in the body. A commit that changes a
  noise parameterisation or an RNG stream name is not a `chore`.

## Debugging

Report counters before reading code:

```
prepared_photons   no_emission_slots
channel_delivered  channel_lost
detections         failed_reports      unassigned_reports
sifted_bits        estimated_qber
final_keys_equal   alice_abort         bob_abort
```

Save a JSONL trace (`JsonlSink`) when event ordering is unclear.

## Known issues

- `control/` has a 3-file import cycle: `agent.py → context.py → timers.py → agent.py`.
  Don't add to it.
- Not modeled: decoy states, free-space channels, CV-QKD, wavelength multiplexing,
  detector crosstalk, finite-key proofs beyond the BB84 example's teaching budget.
  See `CAPABILITY_MAP.md` §5. Report these as gaps rather than approximating silently.

## Output discipline

- Report every default chosen for an unspecified parameter.
- State approximations and their cost in physical accuracy.
- Never invent an API. If it doesn't exist, say so.
