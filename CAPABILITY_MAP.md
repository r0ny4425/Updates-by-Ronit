# SimYuj Capability Map

**Purpose.** A static lookup table from *network-specification concepts* to *simyuj
APIs and reference implementations*. Read this before exploring the codebase — it
answers "which part of simyuj handles X" without a search.

**Accuracy contract.** This map is a routing index, not an API reference. Class and
parameter names below were taken from working example code, but **always verify the
exact signature** before generating code:

```
graphify explain "<ClassName>"
```

Then read the actual source or the nearest reference notebook. Never emit a call
signature from this file alone.

---

## 0. The architectural rule

```
devices produce reports
agents keep protocol state
agents send public messages
agents schedule local events
the runtime executes everything on one timeline
```

Corollaries that constrain every design decision:

- **Protocol logic lives in `NodeAgent` subclasses.** Never in components, never in `qstate`.
- **Components never call each other directly.** All cross-component work is a scheduled timeline event — including `delay=0` timers and same-tick memory requests.
- **All randomness comes from `Timeline.rng(device_id, component_key, stream_name)`** via named streams. Never `numpy.random` or `random`. All three arguments are required: the device id is what keeps two instances of the same component type from sharing a stream.
- **`qstate` never schedules events or advances time.** Components call into it from event handlers.
- **If one party learns something, there must be a message explaining how.** Do not read another agent's private state.

---

## 1. Reference implementations — pick the nearest one first

Always name a reference before writing code. Copying a working trial is dramatically
more reliable than composing from scratch.

| If the spec looks like… | Copy from | What it demonstrates |
|---|---|---|
| Prepare-and-measure QKD, single photons, 2 nodes | `examples/bb84/` (canonical, most complete) | Source → fiber → detector array, slot timing, sifting, full post-processing |
| Entanglement-based QKD (E91/BBM92) | `tutorials/protocols/e91.ipynb`, `tutorials/four_node_qkd/e91_agents.py` | Entangled-pair source, two-sided measurement, correlation |
| Teleportation / Bell measurement / Pauli correction | `tutorials/protocols/teleportation.ipynb` | Memories + `BellStateAnalyzer` device + `correction_for_bell`, fidelity readout |
| Entanglement purification / distillation | `tutorials/protocols/distillation.ipynb` (BBPSSW) | Two stored pairs, bilateral CNOT, sacrificial Z measurement, keep/reject |
| Multi-node (3+), concurrent sessions, routing | `tutorials/four_node_qkd/` | Multiple protocols on one network, per-session bookkeeping |
| Entanglement staged for later consumption | `tutorials/11_capstone/01_prepare_entanglement_for_teleportation.ipynb` | Pair registry + resource reservation + memory lifecycle |
| Classical post-processing only | `examples/postprocessing/bb84_event/` | Sifting → QBER → Cascade → verification → privacy amplification |

**Repeater chains / entanglement swapping** have no single reference notebook. Compose
from: teleportation (BSA + corrections) + distillation (memory pair handling) +
four_node_qkd (multi-hop topology) + `entanglement/` registry + `metrics/` route scoring.
Flag this to the user as a compose-from-parts case, not a copy case.

---

## 2. Subsystem labs — read when a stage is unfamiliar

Each lab is a focused notebook. Read one lab, not the whole tutorial set.

| Lab | Covers |
|---|---|
| `01_engine` | Events, same-time ordering, batch closure, cancel/reschedule, named RNG streams |
| `02_tracing` | Log levels, TextSink/JSONL sinks, timeline logging, trace reading |
| `03_runtime` | `BindingContext`, `bind_many`, why late RNG setup fails, `SessionRuntime` lifecycle |
| `04_qstate` | Refs and records, gates, Bell pairs, density + noise, POVM, Bell-diagonal states |
| `05_components/01` | Ports, wiring, port kinds, absolute delivery time |
| `05_components/02` | Single-photon and entangled-pair sources |
| `05_components/03` | Classical and quantum channels: delay, loss, jitter, qstate noise |
| `05_components/04` | Detectors and receivers |
| `05_components/05` | Memories: absorb/emit/measure, lifetime, expiry |
| `06_network` | Nodes, links, wires, topology |
| `07_metrics` | Link values, route values, route selection |
| `08_resources` | Memory addresses, slot state, reservations, route requirements |
| `09_entanglement` | Pair registry, lifecycle states, queries, multi-hop routes |
| `10_control` | Agents, timers, messages; memory/resource/pair services; workflow lab |

---

## 3. Spec concept → simyuj module

### 3.1 Photon sources

| Spec says | Module | Notes |
|---|---|---|
| single-photon source, heralded single photon | `simyuj.components.sources.SinglePhotonSource` | `frequency_hz`, `emission_probability`, `wavelength_nm`, `duration_s`, `encoding_scheme`, `sampler`, `timing_profile`. Zero-or-one photon backed by a one-qubit qstate |
| weak coherent pulse, attenuated laser, WCP transmitter | `simyuj.components.sources.WeakCoherentPulseSource` | `device_id`, `frequency_hz`, `intensity`, `encoding_scheme`, `wavelength_nm`, `start_time_s`, `duration_s`, `timing_profile`, `carrier_phase`, `encoding_phase`, `temporal_mode_sigma_s`. Emits `SignalKind.PULSE` carrying a `CoherentState`; **no qstate record is created** |
| entangled pair source, SPDC, EPR source | `simyuj.components.sources.EntangledPairSource` | Two quantum output ports |
| which states are emitted, state distribution | `simyuj.qstate.StateSampler` | `states`, `probabilities`, `rep="ket"`, `labels`. Qstate payloads only — it cannot express a `(mu, phase)` choice |
| per-pulse intensity / phase choice (coherent source) | `simyuj.components.sources.coherent_preparation` | `FixedIntensity`; `FixedCarrierPhase`, `PerPulseRandomCarrierPhase`; `FixedPhase`, `RandomPhaseChoice`, `PhaseSequence`; `DPS_PHASES` |
| optical amplitude value type | `simyuj.primitives.coherent_state.CoherentState` | One stored field `alpha`; `mean_photon_number` and `phase_rad` derived. Not re-exported from `simyuj.signal` |
| emission jitter, timing profile | `simyuj.components.GaussianTiming` | `mean_emission_delay_ticks`, `emission_delay_stddev_ticks`, `max_emission_delay_ticks` |
| encoding basis (polarization, time-bin) | `simyuj.signal.EncodingScheme` | |
| source reports | `SourcePreparationReport`, `CoherentPulsePreparationReport` | Delivered to agent via `AGENT_REPORT` port. The coherent one carries μ, Θ, φ_enc and their alphabet indices, and has no `state_ref` or `sampler_*` |

`WeakCoherentPulseSource` is Alice's **complete** preparation device: it chooses
μ, Θ and φ_enc per pulse and builds `alpha = sqrt(mu) * exp(i*(Theta + phi_enc))`
itself. There is no separate modulator component and none is planned.

**Nothing samples a photon number.** Choosing which μ to prepare is a classical
preparation choice; drawing `n ~ Poisson(mu)` is not, and no code path does it.
Photon statistics enter in closed form at detection.

**Not modeled natively:** decoy states, heralded sources, modulator insertion
loss, finite extinction ratio, laser relative intensity noise, side modes, chirp,
and finite laser linewidth (`FixedCarrierPhase` means infinite coherence length).
Decoy intensity levels are one new selector class in
`sources/coherent_preparation.py` and no other change; the surrounding
photon-number analysis is protocol-level and belongs in the agent.

**The coherent pulse is closed end to end.** `QuantumChannel` branches on the
payload role and carries an amplitude deterministically (§3.2), and
`DelayInterferometer` recombines adjacent pulses and detects both output arms
(§3.3, §3.4). What is still missing on this path is polarization-resolved
detection — a polarizing beamsplitter splits a pulse rather than routing it, and
`polarization_weights` has not shipped — which is what decoy-state BB84 needs.
See section 5 and `docs/dev/dps-design.md`.

### 3.2 Channels

| Spec says | Module | Parameters seen in working code |
|---|---|---|
| optical fiber, quantum channel, lossy link | `simyuj.components.channels.QuantumChannel` | `length_m`, `propagation_speed_m_per_s`, `attenuation_db_per_km`, `fixed_insertion_loss_db`, `timing_jitter_stddev_ticks`, `noise_models=(...)` |
| public/authenticated classical channel | `simyuj.components.channels.ClassicalChannel` | `length_m`, `fiber_speed_m_per_s`, `loss_probability`, `session_id` |
| channel actions | `ACTION_TRANSMIT_QUANTUM`, `ACTION_TRANSMIT_CLASSICAL` | Used as `target_action` in `wire_ports` |

Distance → loss and distance → delay are handled by the channel. Do **not** hand-compute
attenuation in the agent.

`QuantumChannel` handles **two** payload kinds, chosen by the role of a signal's
qstate record (`qstate_payload_role` in `components/quantum_targets.py`), never by
whether it has one:

| Payload | `role` | Loss | Noise |
|---|---|---|---|
| qstate carrier (photon, entangled member) | `"qubit"` | Bernoulli trial at `10**(-L/10)`; discarded on loss | `noise_models` via Kraus |
| coherent pulse | `None` | `alpha -> sqrt(eta)*alpha`, deterministic, nothing discarded | `noise_models` **rejected** — use `phase_noise_stddev_rad` |
| polarized coherent pulse | `"mode"` | as above; the record is **not** discarded | `noise_models` via Kraus, on the record |

`eta` is one fibre property with two correct consequences; there is no second loss
field. On the amplitude path `lost_count` stays **0** and
`delivered_count == received_count` however lossy the fibre — read
`attenuated_count` instead, and never read `channel_lost == 0` as "lossless".
The loss RNG is never consumed there, so an all-amplitude run replays identically
at any seed.

`phase_noise_stddev_rad` (default `0.0`) applies a per-pulse optical phase shift.
`timing_jitter_stddev_ticks` must be zero for pulses and `noise_models` must be
empty unless a `"mode"` record is present; both are rejected at **event** time,
because a channel cannot know at construction what it will carry. A fibre
configured for BB84 therefore cannot be reused unchanged for DPS.

Optical arithmetic lives in `components/coherent_optics.py` — `attenuated` and
`phase_shifted` today; `split_50_50`, `interfere`, `gaussian_temporal_overlap`
and `click_probability` ship with their first callers.

**Not modeled natively:** free-space/atmospheric channels, wavelength-dependent
loss, polarization-mode dispersion, active channel drift, optical gain
(`attenuated` rejects a power transmission above 1), chromatic dispersion and
pulse broadening (`temporal_mode_sigma_s` is unchanged in flight). **Channel
phase noise is independent per pulse**, which is pessimistic: real fibre phase
noise is strongly correlated over a nanosecond slot, so an IID draw gives the
*differential* phase a variance of `2*sigma**2` and reads a phase-encoded
protocol's QBER high. Report the discrepancy; never tune `sigma_phi` down to hide
it. Approximate with the existing loss + noise_models, and say so explicitly in
the report.

### 3.3 Modulators and interferometers

| Spec says | Module | Parameters |
|---|---|---|
| delay-line / unbalanced Mach-Zehnder interferometer, DPS receiver | `DelayInterferometer` in `simyuj.components.interferometers` | `detectors` (**required**), `delay_s` **or** `delay_ticks` (exactly one, and it is tau), `short_delay_ticks`, `click_resolver`, `detection_window_ticks`, `gate_model`, `flush_priority` |
| 50:50 beamsplitter, interference, pulse-envelope overlap | `components/coherent_optics.py` | `split_50_50`, `interfere`, `gaussian_temporal_overlap` |
| coherent-pulse click probability | `components/coherent_optics.py` | `click_probability(mu, efficiency=eta)` = `1 - exp(-eta*mu)` |

`DelayInterferometer` has one quantum ingress port, **two** quantum egress ports
(`out_0`, `out_1`, both of which must be connected), a classical `report` port
carrying `InterferenceReport`, and a classical `detection` port carrying
`DetectionReport`.

**Its optics are ideal; its two internal detectors are not.** A DPS receiver is
one physical unit, so detection lives inside rather than downstream — which also
means the two BS2 outputs become one slot decision with no arrival buffering.
The `detectors` parameter is required and takes exactly two
`SinglePhotonDetector` channels, index 0 reading `out_0`; `detectors=None` is a
supported, explicit choice that reproduces the pre-detection component exactly,
including declaring no RNG stream.

Read these before wiring one:

- An **N-pulse train gives N+1 output slots**. The first pulse's short arm and
  the last pulse's long arm each meet vacuum and carry no bit.
- **Nearest-neighbour pairing only**, so `tau` approximately equal to the pulse
  period is the supported regime. `tau = 2T` is a gap, not a configuration.
- The run must reach `last_arrival + 2*tau + 1`, or the final slots never
  execute. `Timeline.run_until_empty()` does this.
- It never validates `tau` against the pulse period — it cannot, it never sees
  the clock. A mismatch appears as `temporal_overlap` collapsing on every slot.
- `short_delay_ticks` is the **common** transit both arms take; `delay_ticks` is
  the **extra** time the long arm takes. Only the difference is physical, so a
  common transit shifts every downstream tick and changes no interference.
- Both ports are evaluated independently, so **a double click from two signal
  clicks is a real event at a real rate** — unlike `DetectorArray`, where a
  readout exposes one detector per outcome and that rate is identically zero.
  `ThresholdClickResolver(double_click_policy=...)` decides what the slot then
  reports; the rate is physics and the response is protocol.
- The two edge slots split `mu/4` to each port and carry no bit. They still get
  a report — dropping them is the agent's job, on the `None` pulse index in the
  report metadata.
- A `DetectionReport` from here carries `measurement_label=None` and
  `signal_id=None`: nothing was measured, and two signals left BS2 rather than
  one. `outcome` is the port name, and `meta` carries `interference_index` — the
  join key back to the `InterferenceReport` — plus mu, phase, overlap and both
  pulse indices.

**No phase modulator exists and none is planned; that is not the gap.**
`WeakCoherentPulseSource` (§3.1) chooses the encoding phase itself, so a separate
modulator would add an event hop and a report describing a phase the source
already knows. The receiver piece still missing is **polarization-resolved
detection**, not a modulator — see §5.

`examples/dps/trial.py` runs the whole chain — source -> channel ->
interferometer -> taps — and checks that the bits read off the output ports match
the bits Alice prepared. It still runs `detectors=None` and reads its bits from
the reported intensities: turning that example over to real detection needs an
agent to consume the reports, and is not done. It is also the only end-to-end exercise of
`QuantumChannel`'s coherent-amplitude path, and its CLI shows the two outcomes
that differ: attenuation costs signal and no key, per-pulse phase noise costs
key.

### 3.4 Detectors

| Spec says | Module | Parameters |
|---|---|---|
| SPD, APD, SNSPD, detector efficiency | `SinglePhotonDetector` + `SinglePhotonDetectorParams` | `efficiency`, `dark_count_rate_hz`, `dead_time_ticks`, `jitter_stddev_ticks`, `p_afterpulse`, `afterpulse_decay_ticks`, `photon_number_resolving` |
| basis choice, measurement settings | `DetectorArray(measurement=Measure.random(...))` or `Measure.basis("z")` | Weighted random basis choice supported |
| which detector fires for which outcome | `DetectorArray(readout={...})` | e.g. `{"Z": {"0": "d_z0", "1": "d_z1"}}` |
| double clicks, click discrimination | `ThresholdClickResolver(double_click_policy=...)` | From `detectors.primitives.click` |
| detection gate / coincidence window | `detection_window_ticks`, `detectors.primitives.window`, `.gate` | |
| coherent-pulse / weak-laser detection | `DelayInterferometer(detectors=...)` (§3.3) | Two channels, `1 - exp(-eta*mu)` per port, genuine double clicks |
| detection probability that is not a bare efficiency | `DetectorExposure(signal_click_probability=...)` | **Overrides** `params.efficiency`, never multiplies it |
| Bell-state measurement, BSM, swapping node | `BellStateAnalyzer` + `ACTION_RUN_BELL_ANALYSIS`, `BSMModel` | Real device event, not a direct qstate call |
| direct qubit readout (no photon) | `simyuj.components.detectors.qubit_readout` | |
| detector output | `DetectionReport`, `ACTION_DETECT_SIGNAL` | |

Detector primitives live in `detectors/primitives/`: `params`, `measurement`, `readout`,
`click`, `dark_counts`, `gate`, `window`, `rng`, `reports`, `result_labels`, `actions`.

`SinglePhotonDetector` is **not** a `Component`: no ports, no timeline, no
qstate. It is composed into event-facing components — `DetectorArray`,
`BellStateAnalyzer`, `DelayInterferometer` — which own the ports and the RNG
streams. A new receiver should compose it too, not subclass a component built
around a qstate spine.

**`efficiency` is not "probability of a click".** For a qubit carrier it is, but
for a coherent pulse the click probability is `1 - exp(-eta*mu)`, so a perfect
detector at `mu = 0.2` still sees nothing 82% of the time — most pulses contain
no photon at all. When an exposure carries `signal_click_probability`, that value
*replaces* `params.efficiency` for the signal click: `eta` is already inside the
exponent, and multiplying would apply it twice and lower every rate by that
factor with nothing raising. Dark counts and afterpulses always read `params`.

### 3.5 Quantum memories

| Spec says | Module | Parameters |
|---|---|---|
| quantum memory, storage, register | `simyuj.components.memories.QuantumMemory` | `memory_id`, `num_positions`, `absorb_delay_ticks`, `emit_delay_ticks`, `measure_delay_ticks`, `recovery_ticks`, `storage_lifetime_ticks`, `noise_models=(...)` |
| coherence time, memory decay | `storage_lifetime_ticks` + `noise_models` | Time-dependent noise via `qstate.noise.time` / `t1t2` |
| store / retrieve / measure | `MEMORY_ABSORB`, `MEMORY_EMIT`, `MemoryMeasureRequest` | All are scheduled events |
| memory reports | `MemoryAbsorbReport`, `MemoryEmitReport`, `MemoryMeasurementReport`, `MemoryOperatorReport`, `MemoryDiscardReport` | |
| slot state | `MemoryPositionRecord`, `MemoryPositionStatus`, `memory_subsystem_id` | |
| agent-side access | `ctx.memory` (`MemoryService`) | Schedules requests at current tick |

### 3.6 Noise models

| Spec says | API (`simyuj.qstate.noise`) |
|---|---|
| depolarizing noise | `depolarizing(p)`, `DepolarizingNoise`, `two_qubit_depolarizing` |
| amplitude damping, T1, relaxation | `amplitude_damping(...)`, `t1t2` module |
| dephasing, phase flip, T2 | `phase_flip(...)`, `dephase` module |
| Pauli noise | `pauli` module |
| arbitrary Kraus channel | `kraus` module |
| noisy gate operations | `noisy_gates` module |
| time-dependent noise | `time` module |

Noise attaches at the **component** level via `noise_models=(...)` on channels and
memories. It is *not* applied by the agent.

**Noise policy:** `QuantumStateManager` defaults to exact evolution (ket records convert
to density before noise). For pure-trajectory performance use
`noise_mode="sampled_ket"` with an explicit noise RNG. Density records stay exact;
Bell-diagonal records stay compact for supported Pauli noise.

### 3.7 Quantum state operations

| Spec says | Module |
|---|---|
| gates (H, X, Z, S, CNOT, CZ) | `simyuj.qstate.ops` — `H`, `I`, `S`, `X`, `Z`, `unitary` |
| multi-controlled gates | `ops.gates` — `MCX`, `MCZ` |
| rotations | `ops.rotations` |
| reset / discard | `ops.reset` |
| Pauli frame tracking | `ops.frame` |
| Bell correction after BSM | `ops.correction_for_bell` |
| fidelity, state metrics | `qstate.state.metric` — `fidelity` |
| Bell states | `qstate.measure.bell` — `bell_vector`, `bell_vectors`, `bell_density_matrix`. `BellDiagState` is the one that lives in `qstate.state` |
| basis states | `qstate.state` — `zero`, `plus_i`, etc. |
| projective measurement | `qstate.measure.projective`, `.basis`, `.result` |
| POVM | `qstate.measure.povm` |
| Bell measurement (math level) | `qstate.measure.bell`, `BellResult` |
| partial trace / reduced state | `qstate.state.reduce` |
| representation conversion | `qstate.state.convert` (ket / density / bell_diag) |
| subsystem naming and layout | `qstate.space` — `dim`, `layout`, `subsystem`, `target`; `SubsystemId` |
| state store and manager | `QuantumStateManager`, `QuantumStateStore`, `StateRef` |
| invariant checking (debug) | `qstate.debug.invariant`, `qstate.debug.dump`, `qstate.check` |

### 3.8 Network topology

| Spec says | API (`simyuj.network`) |
|---|---|
| node, station, repeater site | `Node("id")`, `network.add_node(node)` |
| attach hardware to a node | `node.add_device("name", device)` |
| attach protocol logic to a node | `node.add_agent(agent)` |
| quantum link between nodes | `network.add_quantum_link(name, a, b, channel=...)` |
| classical link between nodes | `network.add_classical_link(name, a, b, channel=...)` |
| exact port-to-port delivery | `network.wire_ports(name, out_port, in_port, target_action=...)` |
| routing, path finding | `network.routing`, `network.planning`, `network.topology` |

Links declare topology; **wires declare exactly which output port delivers to which
input port**. Both are required. Keep wiring explicit and in one file.

### 3.9 Protocol control

| Spec says | API (`simyuj.control`) |
|---|---|
| protocol state machine | Subclass `NodeAgent` |
| start behavior | `on_start(start, ctx)` |
| react to device output | `on_report(report, ctx)` |
| react to peer messages | `on_message(message, ctx)` |
| local scheduled steps | `on_event(event, ctx)` |
| timeouts, retries, deadlines | `on_timer(timer, ctx)` + `ctx.timers` (supports `replace=`, `set_once=`, cancel by id) |
| send classical message | `agent.enable_classical()`, `endpoint.add_route(peer_agent_id, port_name)`, `ctx.classical` |
| resolve node-local devices | `ctx.devices` |
| memory operations | `ctx.memory` |
| reserve memory slots | `ctx.resources` (`ResourceManager` behind it) |
| entangled pair bookkeeping | `ctx.pairs` (`EntangledPairRegistry` behind it) |
| run the simulation | `SessionRuntime(timeline=..., network=...).run()` |
| port actions | `AGENT_REPORT`, `AGENT_MESSAGE` |

Agents are discovered from nodes; the runtime binds devices before agents and schedules
one `agent_start` per agent in sorted `agent_id` order for deterministic replay.

**Known wart:** `control/` contains a 3-file import cycle
(`agent.py → context.py → timers.py → agent.py`). Don't add to it.

### 3.10 Entanglement and resources

| Spec says | Module |
|---|---|
| track entangled pairs, lifecycle | `simyuj.entanglement` — `registry`, `pair`, `queries`, `build` |
| pair states (available/reserved/consumed/expired/failed) | `entanglement.registry` lifecycle transitions |
| memory slot reservation | `simyuj.resources` — `ResourceManager`, `reservation`, `memory` |
| route requirements | `resources.route_requirements` |
| link quality, route scoring | `simyuj.metrics` — `link.py`, `path.py` |

### 3.11 Engine, timing, randomness

| Spec says | API |
|---|---|
| simulation clock, reproducibility | `Timeline(master_seed=N)` |
| schedule work | `Timeline.schedule(Event(time=, target_ref=, action=, payload_ref=))` |
| run modes | `run()`, `run_until(t)`, `run_one_step()` |
| named random stream | `Timeline.rng(device_id, component_key, stream_name)` — e.g. `timeline.rng("alice_mod", "phase_modulator", "phase")` |
| seconds ↔ ticks | `simyuj.primitives.units.seconds_to_ticks`, `ticks_to_seconds` |
| event ordering | `engine.event_ordering` |
| run statistics | `engine.execution_summary`, `timeline_statistics` |

`run_until(t)` only advances already-scheduled events — it does not bind or schedule
agent starts. Use `runtime.run()` for the normal lifecycle.

### 3.12 Signals, messages, identifiers

| Spec says | API |
|---|---|
| a photon / quantum carrier in transit | `simyuj.signal.Signal`, `SignalKind`, `EncodingScheme` |
| classical protocol message | `simyuj.primitives.messages.transport.ClassicalMessage` |
| subsystem identity | `simyuj.primitives.subsystems.SubsystemHandle`, `qstate.SubsystemId` |
| id validation, metadata | `primitives.ids`, `primitives.meta`, `primitives.validation` |

### 3.13 Observability

| Spec says | API (`simyuj.tracing`) |
|---|---|
| event log, run trace | `SimulationLogger`, `NullLogger` |
| log verbosity | `LogLevel` — an `IntEnum`, least to most verbose: `OFF`(0), `ERROR`, `WARNING`, `INFO`, `DEBUG`, `TRACE`(5). A logger at a level emits that level and every lower one. **Most component records are `DEBUG`** — `signal_forwarded`, source `emit` — so a run traced at `INFO` shows almost no device activity |
| persist a run | `JsonlSink` |
| human-readable output | TextSink |

Logging is observational and must never change the run.

---

## 4. Standard debug counters

When a run looks wrong, report these before reading code (names from the BB84 example;
adapt per protocol):

```
prepared_photons        no_emission_slots
channel_delivered       channel_lost
bob_detections          bob_failed_reports      bob_unassigned_reports
sifted_bits             estimated_qber
final_keys_equal        alice_abort             bob_abort
```

Save a JSONL event log when ordering is unclear.

---

## 5. Known gaps — say so, don't fake it

Not natively modeled. If the spec requires one, report it as a gap and propose either an
approximation or a new component:

- **Photon arrival time within the pulse envelope.** A click is reported at the
  detector window's start tick plus jitter, never at a position sampled inside
  the Gaussian the pulse actually occupies. Anything that needs sub-pulse
  timing resolution — time-bin discrimination finer than a slot, a COW monitor
  line, a jitter-limited QBER budget — is therefore not modelled.

  This is blocked on a real defect rather than merely unwritten. Every timing
  term in `SinglePhotonDetector` is currently non-negative (`_apply_jitter`
  clamps at `max(0, ...)`), which makes the window filter's lower bound at
  `single_photon.py:193` dead code and the whole-window dead-time gate at `:174`
  conservative-only. An arrival offset is **signed** — the envelope is symmetric
  about the pulse tick, by the contract on `Signal.temporal_mode_sigma_s` — so
  adding one would make that filter silently discard early-arriving photons,
  about half the envelope, with no error and no symptom beyond an efficiency
  quietly too low. There is a second pre/post-jitter mix in the afterpulse model
  (`elapsed_ticks < 0` returns `None` silently). Fix the asymmetry first, with a
  fingerprint test, then add the term.
- **Envelope shape out of an interferometer.** At `|gamma| < 1` each output port
  carries an interfering component plus an orthogonal residual: the true
  envelope is a superposition of two Gaussians `delta_ticks` apart, and the
  single `temporal_mode_sigma_s` the port reports describes neither. The mean
  photon number is exact and a threshold detector reading intensity is
  unaffected, which is why detection is sound today — but this is the reason the
  arrival-sampling gap above cannot simply be closed at the detector.
- **Polarization-resolved detection, and therefore decoy-state BB84.** A weak
  coherent pulse hitting a polarizing beamsplitter *splits*: `mu_H = mu*w_H` and
  `mu_V = mu*w_V` reach two detectors independently, and the resulting
  double-click rate is basis-dependent and physically meaningful. Measuring the
  polarization qstate and routing the whole pulse to one detector is the
  single-photon model and would make the double-click rate identically zero at
  every `mu` — the guard at `quantum_targets.qubit_carrier_targets_from_signal`
  exists to stop exactly that. `polarization_weights` and
  `rotated_polarization` have not shipped. See `docs/dev/dps-design.md` §8.
- **Photon-number resolution on a detected signal.** A signal contributes at
  most one Bernoulli candidate per port, so `P(n >= 2 | click)` is not modelled
  even at `photon_number_resolving=True` — that flag only governs whether dark
  counts and afterpulses can add further clicks in one window.
- **Interferometer non-idealities.** `DelayInterferometer`'s optics are ideal:
  no insertion loss, no arm imbalance, no splitting-ratio error, no internal
  phase noise, and no thermal or mechanical drift of the arm lengths. Every
  optical imperfection must arrive with the incoming pulses. Pairing beyond
  nearest neighbour (`tau = 2T` and up) needs a keyed queue and a different
  component; it is a gap, not a configuration.
- Free-space / satellite channels, atmospheric turbulence
- Continuous-variable QKD (repo is discrete-variable)
- Wavelength multiplexing, frequency conversion
- Active feedback / drift compensation
- Chromatic dispersion and pulse broadening
- Nonlinear fiber effects (SPM, XPM, four-wave mixing)
- Frequency-dependent loss; polarization-mode dispersion
- Channel phase drift of any kind, correlated or independent
- Optical gain / amplification
- Detector crosstalk between array elements
- Finite-key security proofs beyond the teaching-level budget in the BB84 example
