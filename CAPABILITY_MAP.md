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
| WCP, weak coherent pulse, attenuated laser | `simyuj.components.sources.WeakCoherentPulseSource` | `frequency_hz`, `mean_photon_number`, `wavelength_nm`, `duration_s`, `encoding_scheme`, `timing_profile`. Non-sampling: every active slot emits one pulse carrying the same `CoherentState`, and no qstate record is created |
| entangled pair source, SPDC, EPR source | `simyuj.components.sources.EntangledPairSource` | Two quantum output ports |
| optical amplitude of a pulse | `simyuj.signal.CoherentState` | `alpha` is the only stored field; `mean_photon_number` and `phase_rad` are derived. `from_mean_photon_number(mu, phase_rad=)`, `with_phase_shift(theta)`, `attenuated(power_transmission)` |
| which states are emitted, state distribution | `simyuj.qstate.StateSampler` | `states`, `probabilities`, `rep="ket"`, `labels` |
| emission jitter, timing profile | `simyuj.components.GaussianTiming` | `mean_emission_delay_ticks`, `emission_delay_stddev_ticks`, `max_emission_delay_ticks` |
| encoding basis (polarization, time-bin) | `simyuj.signal.EncodingScheme` | |
| source reports | `SourcePreparationReport` | Delivered to agent via `AGENT_REPORT` port |

**Not modeled natively:** decoy states, multi-photon/photon-number statistics beyond
`emission_probability`, heralded sources. These are protocol-level additions —
implement in the agent, not the component, unless a real new device is required.

### 3.2 Channels

| Spec says | Module | Parameters seen in working code |
|---|---|---|
| optical fiber, quantum channel, lossy link | `simyuj.components.channels.QuantumChannel` | `length_m`, `propagation_speed_m_per_s`, `attenuation_db_per_km`, `fixed_insertion_loss_db`, `timing_jitter_stddev_ticks`, `noise_models=(...)` |
| coherent-pulse transport, phase-encoded fiber | the same `QuantumChannel` | adds `phase_noise_stddev_rad`. Do **not** build a parallel classical channel |
| public/authenticated classical channel | `simyuj.components.channels.ClassicalChannel` | `length_m`, `fiber_speed_m_per_s`, `loss_probability`, `session_id` |
| channel actions | `ACTION_TRANSMIT_QUANTUM`, `ACTION_TRANSMIT_CLASSICAL` | Used as `target_action` in `wire_ports` |

Distance → loss and distance → delay are handled by the channel. Do **not** hand-compute
attenuation in the agent.

**The channel branches on payload.** A qstate-backed signal takes the Bernoulli
survival path; a signal carrying a `coherent_state` takes a deterministic one:
`α → √η·α·e^{iδφ}` at a fixed delay, where η is the *same* power transmission
(`survival_probability`) the qubit path uses as a survival probability. One
physical property of the fiber, two correct consequences. Carrying things is a
transport component's job, so it accepts both — unlike a transform component such
as `PhaseModulator`, which rejects what it cannot transform.

Consequences worth knowing before reading a coherent run's counters:

- μ scales as η, not √η. μ=0.2 through η=0.1 is μ=0.02.
- **Nothing is discarded**, so `lost_count` is always 0 and `delivered_count`
  equals `received_count` however lossy the fiber is. Loss appears as reduced μ;
  the `pulse_forwarded` log record carries `mean_photon_number_in` and
  `mean_photon_number_out`. Reading `channel_lost == 0` as "lossless" is a trap.
- Total attenuation **delivers coherent vacuum**, it does not drop the pulse.
  Deciding no photon was seen is the detector's job.
- The loss RNG stream is never consumed on this path. Attenuation is arithmetic,
  not sampling.
- Metadata is `channel_power_transmission`, not `survival_probability` — a pulse
  that never faced a Bernoulli trial must not carry a record claiming it did.
- Adjacent-pulse spacing is preserved (fixed delay), which is what a delay
  interferometer downstream needs.

Three configurations are **rejected** rather than silently ignored for a coherent
pulse: `timing_jitter_stddev_ticks` (independent per-pulse jitter destroys pulse
spacing and can reorder pulses), `noise_models` (Kraus operators on qubit axes
have no representation for an optical amplitude), and a signal carrying both
`state_ref` and `coherent_state`.

**Not modeled natively:** free-space/atmospheric channels, wavelength-dependent loss,
polarization-mode dispersion, active channel drift. Approximate with the existing
loss + noise_models, and say so explicitly in the report. For coherent pulses see
section 5 — the phase-noise model in particular is independent per pulse, which is
pessimistic for closely spaced pulses.

### 3.3 Modulators

| Spec says | Module | Parameters |
|---|---|---|
| phase modulator, phase encoder, DPS/DQPS encoder | `simyuj.components.modulators.PhaseModulator` | `device_id`, `phase_selector`. Applies `alpha -> alpha * exp(i*theta)`; lossless, instantaneous, no qstate |
| which phase per pulse | `FixedPhase`, `RandomPhaseChoice`, `PhaseSequence` | Frozen strategy objects. Default is `RandomPhaseChoice(phases=DPS_PHASES)`, uniform over `(0, pi)` |
| modulator actions | `ACTION_MODULATE_PHASE` | Used as `target_action` in `connect_ports` / `wire_ports` |
| modulator output | `PhaseModulationReport` | Carries the *requested* `phase_rad`, `phase_index`, `modulation_index`, and the incoming `pulse_index`. No label field: a phase is fully described by its value and index |

A modulator transforms a signal in flight: one input port, one output port, no
self-scheduled emissions, and signal identity preserved across the transform. A
signal arriving without a `coherent_state` is rejected, not passed through.

The applied `theta` is **not** recoverable from the amplitude afterwards.
`CoherentState.phase_rad` is the total wrapped phase; read the report or the
signal metadata for what the modulator actually imposed.

**Not modeled natively:** insertion loss, finite extinction ratio, modulation
bandwidth / rise time, residual chirp. See section 5.

### 3.3b Interferometers

| Spec says | Module | Parameters |
|---|---|---|
| delay interferometer, unbalanced Mach-Zehnder, DPS/DQPS receiver | `simyuj.components.interferometers.DelayInterferometer` | `device_id`, and **exactly one** of `delay_s` / `delay_ticks`, plus `flush_priority` |
| interferometer actions | `ACTION_INTERFERE` (ingress), `ACTION_RESOLVE_BS2`, `ACTION_FLUSH_DELAY_ARM` (both self-scheduled) | `ACTION_INTERFERE` is the `target_action` in `connect_ports` |
| interferometer output | two quantum egress ports, `out_0` and `out_1` | The only component in the repo with two quantum output ports. **Both must be connected.** `out_0` is the destructive port when the arms are in phase |
| what happened in a slot | `InterferenceReport` | `temporal_overlap`, `delta_ticks`, both BS2 ticks, `mean_photon_number_in/_0/_1`, and `short_pulse_index` / `long_pulse_index` |
| beamsplitter arithmetic | `signal.optical.split_50_50`, `interfere`, `gaussian_temporal_overlap` | Module-level functions; unit-testable without a timeline |
| pulse temporal envelope | `Signal.temporal_mode_sigma_s`, `WeakCoherentPulseSource(temporal_mode_sigma_s=...)` | Seconds. **Field**-envelope standard deviation |

**One equation, no decision tree.** Every recombination is

```
mu_0 = ½[|α_s|² + |α_l|² − 2·Re(α_s* · α_l · γ)]
mu_1 = ½[|α_s|² + |α_l|² + 2·Re(α_s* · α_l · γ)]
```

Vacuum inputs, `γ = 0`, unequal amplitudes, the first pulse, and the last pulse
are values of that equation, not branches around it — a vacuum arm kills the
interference term for any `γ`, because the term is proportional to both
amplitudes. `mu_0 + mu_1 = mu_s + mu_l` holds exactly at every `γ`; that is the
invariant that catches a convention error, and every test asserts it.

**Beamsplitter convention: the real 50:50 matrix `(1/√2)[[1,−1],[1,1]]` at both
splitters**, stated once at the top of `signal/optical.py` and used everywhere
including the tests. The symmetric `[[1,i],[i,1]]` convention is the same
physical device and differs only by an unobservable global phase on `out_1`; the
real one is chosen because it puts the interference term in `Re` rather than
`Im`, matching the equations above.

**Where γ comes from.** An amplitude does not say *when* the light is, so the
temporal envelope travels separately on `Signal.temporal_mode_sigma_s` — a mode
property, so it sits beside `wavelength_nm` rather than inside `CoherentState`,
and `PhaseModulator` and `QuantumChannel` carry it through with no code of their
own. σ is defined by `f(t) = (πσ²)^(−1/4)·exp[−(t−t₀)²/2σ²]` with `∫|f|² = 1`,
so it is the **field** envelope's standard deviation; the intensity FWHM is
`2√(ln2)·σ ≈ 1.665σ`. For equal widths `γ = exp(−Δt²/4σ²)` — an
*intensity*-envelope σ would put an 8 in that denominator, and the two must not
be mixed. A pulse with `temporal_mode_sigma_s=None` is **rejected**, by the same
transform-rejects rule as `PhaseModulator`.

**Timing is observed, never corrected.** BS1 acts at the actual arrival tick, so
a late pulse interferes less rather than being realigned. The component never
validates τ against the pulse period: it does not know the period, and taking it
as configuration would put the source's `frequency_hz` in two places.

Consequences worth knowing before reading a run's output:

- **An N-pulse train gives N+1 output slots.** The first pulse's short arm and
  the last pulse's long arm each meet vacuum and split 50:50, carrying no bit.
- **Nearest-neighbour only.** One contribution is held at a time, so each pulse
  pairs with its immediate predecessor. τ ≈ T is the intended regime; τ = 2T is
  *not* supported — see section 5.
- **The flush deadline is that assumption, not a decay estimate.** A held arm is
  combined with vacuum at `arrival + 2τ`. At σ comparable to τ the discarded
  overlap is still ~0.78. A pulse arriving at exactly `arrival + 2τ` pairs; one
  tick later it does not.
- `flush_priority` (default 10000) must stay strictly above the upstream
  `delivery_priority` (default 0), or the boundary above inverts.
- **Outputs are new optical events** with new ids, like a memory re-emission and
  unlike a modulator transform. Both contributing `pulse_index` values travel in
  the metadata and the report; `None` there means that arm was vacuum.
- **Outputs are intensity-exact and mode-truncated.** At |γ| < 1 the port field
  is a superposition of two non-identical envelopes. `mean_photon_number` is
  exact; the phase is the interfering component's and the width is the short
  arm's, and **neither may be used for a further phase-sensitive or
  temporal-mode interference**. At |γ| = 1 all three are exact.
- **No RNG streams at all.** The device is ideal by specification, so `bind`
  declares none rather than declaring one that is never consumed.
- A run must reach `last_arrival + 2τ`, or the final slot never executes.

**Not modeled natively:** insertion loss, arm imbalance, non-ideal splitting
ratio, arm-length drift, polarization mismatch between arms. See section 5.

### 3.4 Detectors

| Spec says | Module | Parameters |
|---|---|---|
| SPD, APD, SNSPD, detector efficiency | `SinglePhotonDetector` + `SinglePhotonDetectorParams` | `efficiency`, `dark_count_rate_hz`, `dead_time_ticks`, `jitter_stddev_ticks`, `p_afterpulse`, `afterpulse_decay_ticks`, `photon_number_resolving` |
| basis choice, measurement settings | `DetectorArray(measurement=Measure.random(...))` or `Measure.basis("z")` | Weighted random basis choice supported |
| which detector fires for which outcome | `DetectorArray(readout={...})` | e.g. `{"Z": {"0": "d_z0", "1": "d_z1"}}` |
| double clicks, click discrimination | `ThresholdClickResolver(double_click_policy=...)` | From `detectors.primitives.click` |
| detection gate / coincidence window | `detection_window_ticks`, `detectors.primitives.window`, `.gate` | |
| Bell-state measurement, BSM, swapping node | `BellStateAnalyzer` + `ACTION_RUN_BELL_ANALYSIS`, `BSMModel` | Real device event, not a direct qstate call |
| direct qubit readout (no photon) | `simyuj.components.detectors.qubit_readout` | |
| detector output | `DetectionReport`, `ACTION_DETECT_SIGNAL` | |

Detector primitives live in `detectors/primitives/`: `params`, `measurement`, `readout`,
`click`, `dark_counts`, `gate`, `window`, `rng`, `reports`, `result_labels`, `actions`.

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
| Bell states | `qstate.state` — `bell_vector`, `bell_density_matrix`, `BellDiagState` |
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
| log verbosity | `LogLevel` (INFO / TRACE) |
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

- Decoy-state BB84 (photon-number statistics)
- Free-space / satellite channels, atmospheric turbulence
- Continuous-variable QKD (repo is discrete-variable)
- Wavelength multiplexing, frequency conversion
- Active feedback / drift compensation
- Modulator insertion loss (the phase modulator is lossless)
- Modulator finite extinction ratio (an imposed phase is exact)
- Modulator bandwidth / rise time (modulation is instantaneous)
- Modulator residual chirp (phase modulation does not disturb amplitude)
- Chromatic dispersion and pulse broadening for coherent pulses. A pulse's
  temporal envelope *is* now described, by `Signal.temporal_mode_sigma_s`, but
  nothing broadens it in flight: the channel carries the field through
  unchanged, so a long fiber does not reduce interference visibility the way a
  real one would
- Non-Gaussian pulse envelopes (`gaussian_temporal_overlap` is the only overlap
  model; a sech² or square envelope would need its own closed form)
- Delay-interferometer insertion loss, arm imbalance, and non-ideal splitting
  ratio (the device is ideal; every imperfection must arrive with the pulses)
- Delay-interferometer arm-length drift, thermal or mechanical (τ is fixed for
  the whole run, and there is no internal phase noise)
- Polarization mismatch between interferometer arms (the modelled overlap γ is
  purely temporal)
- **τ ≠ T_pulse for the delay interferometer.** One long-arm contribution is
  held at a time, so pairing is nearest-neighbour only. A τ of two slot periods
  would need pulse *k* to meet pulse *k+2*, but pulse *k+1* arrives first and
  takes the holder. Arbitrary τ needs a keyed queue and belongs in a separate
  component; report it as a gap rather than configuring τ = 2T and reading the
  result
- Phase-sensitive or temporal-mode use of a delay-interferometer output at
  |γ| < 1. The port field is then a superposition of two non-identical
  envelopes; the emitted signal keeps the exact mean photon number but its
  phase and width are a truncation. Chaining a second interferometer, or any
  downstream device that treats those as physical, is outside the model
- Nonlinear fiber effects on coherent pulses (SPM, XPM, four-wave mixing)
- Frequency-dependent loss and polarization for coherent pulses
- Per-pulse timing jitter for coherent pulses (rejected, not modelled — it
  destroys the adjacent-pulse spacing interference depends on)
- Correlated channel phase drift. `phase_noise_stddev_rad` draws **independently
  per pulse**, so the differential phase between adjacent pulses has variance
  `2σ_φ²`. Real fiber phase noise over a slot period of order a nanosecond is
  strongly correlated between neighbours, so this model *over-estimates*
  differential-phase error — a phase-encoded protocol's QBER will read
  pessimistic at any given σ_φ. A Wiener/Ornstein-Uhlenbeck drift is a different
  model with its own state; report the discrepancy rather than tuning σ_φ to hide it
- Qstate noise applied to a coherent pulse (rejected: Kraus operators are shaped
  `(2**arity, 2**arity)` and have no representation for an optical amplitude)
- Optical gain / amplification (`attenuated` rejects η > 1)
- Detector crosstalk between array elements
- Finite-key security proofs beyond the teaching-level budget in the BB84 example
