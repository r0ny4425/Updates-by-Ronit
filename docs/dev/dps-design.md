# DPS-QKD: design document

Status: design only. No source file, test, or other doc is changed by this document.

Scope: the components and shared-code changes needed to run DPS-QKD end to end,
chosen so that COW and decoy-state BB84 can later reuse the same source, channel,
and optical arithmetic without a second round of shared-code churn.

---

## 0. How the claims in this document were verified

**Provenance.** The repo findings, the `Signal`/channel/detector analysis, and the
measurements below were produced in a *different clone* of this repository
(`C:\Simyuj - Fresh (18th August)\Simyuj`) and moved here. That clone's `src/`,
`tests/`, `examples/`, and `tutorials/` are byte-identical to this one apart from
line endings, verified with `diff -r --strip-trailing-cr`, so **every path, line
number, and API claim in sections 1 through 10 applies unchanged**. Eight citations
were re-checked against this clone directly; one was off by one and is fixed.

**How the exhaustive-search claims were established.** That clone had no
`graphify-out/`, so every "there are exactly N call sites" claim below was
established with ripgrep over `src/`, `tests/`, and `examples/`, and each one
states the query that produced it so it can be re-run — here, against the real
graph, with `graphify explain` and `graphify affected` as a cross-check. Where a
claim could be checked by executing code rather than reading it, it was: the
throughput numbers in section 7 and the `StateSampler` rejection in section 5 are
measured, not inferred.

**The reference implementation has been read.** `scratch/` holds
`weak_coherent_pulse_source.py`, `phase_modulator.py`, `optical.py`,
`delay_interferometer.py`, and `dps-qkd-design-notes.md` — four completed steps,
not a sketch. Section 11 is written against that code, states a reason for every
adoption and every rejection, and is the place to look before re-deciding anything
here. Two of its findings forced corrections to the rest of this document, both
applied and both flagged in place: `_with_metadata` omission raises `AttributeError`
rather than losing data silently (section 1.1, S3), and `CoherentState` as a value
object earns its place (sections 2 and 4, reversing a judgement made without having
seen it). A third finding replaces S9 outright.

**`CAPABILITY_MAP.md` was a live hazard and has been corrected.** It promised
`CoherentState`, `WeakCoherentPulseSource`, `PhaseModulator`,
`DelayInterferometer`, and `Signal.temporal_mode_sigma_s`; searching `src/` and
`tests/` for each of them returns **zero files**. Since `CLAUDE.md` sends every
session there first, sections 3.1, 3.2, 3.3, 3.3b and 5 have been stripped back to
what exists. Section 3's S11 is a record of that edit, not a pending task.

**Baseline confirmed.** `uv run pytest tests/ -q` gives `1460 passed in 24.44s`.

---

## 1. Repo findings

### 1.1 Signal — `src/simyuj/signal/signal.py`

`Signal` is a frozen, `slots=True` dataclass; equality compares every field.
Fields today: `id`, `signal_kind`, `encoding_scheme`, `emission_time`, `origin`,
`wavelength_nm`, `correlation_id`, `correlation_meta`, `state_ref`,
`state_targets`, `protocol_params`, `meta`, `timing_meta`, `validation_flag`.

`validation_flag=False` is a documented fast path: `__post_init__` returns
immediately. `SinglePhotonSource` uses it on the emission hot path.

`SignalKind` has `PHOTON`, `PULSE`, `ENTANGLED_MEMBER`.
**`SignalKind.PULSE` is defined and used nowhere** — searching `src/`, `tests/`,
`examples/` and `tutorials/` for `SignalKind.` returns no `PULSE` hit. It is free
for the WCP.

`EncodingScheme` has `PHASE`, `POLARIZATION`, `FREQUENCY`, `TIME_BIN`.
`EncodingScheme.PHASE` appears exactly once in the repo, in
`tests/primitives/test_messages.py:15`. **Nothing in `src/` branches on
`signal_kind` or `encoding_scheme`** — searching for `signal_kind ==`,
`signal_kind is`, `encoding_scheme ==` and `encoding_scheme is` returns zero hits
in `src/`. Both are descriptive metadata only. This is why "the channel branches
on what a signal *carries*" is implementable without fighting anything: nothing
currently branches on anything else either.

**`Signal._with_metadata` (`signal.py:274`) is the trap this design has to defuse
first.** It rebuilds a `Signal` by copying fourteen fields **one
`object.__setattr__` at a time**, bypassing `__init__` entirely. It has exactly one
caller: `QuantumChannel._with_channel_metadata` (`channels/quantum.py:532`), which
is on the path of every optical pulse here.

A field not added to that method is **left unset**. Because `Signal` is
`slots=True`, it does not fall back to its declared default — the first read raises
`AttributeError: 'Signal' object has no attribute ...`. Confirmed with a minimal
frozen-slots dataclass, and independently recorded in
`scratch/dps-qkd-design-notes.md`. An earlier draft of this document called it
silent data loss; that was wrong. It fails loudly, on the far side of a channel,
at a point far from the edit that caused it. S3 replaces the mechanism rather than
adding three more lines to it — see section 11.2.

`Signal` is constructed in exactly three places in `src/`:
`memories/quantum_memory.py:1467`, `sources/entangled_pair_source.py:593`,
`sources/single_photon_source.py:426`.

### 1.2 Ports and wiring — `components/ports.py`, `components/connections.py`

`Port(name, owner, owner_id, port_kind, direction)`; `PortKind` is
`QUANTUM`/`CLASSICAL`, `PortDirection` is `INGRESS`/`EGRESS`. Ports are structural
only — they never handle events. `connect_ports(source_port, target_port, *,
target_action, connection_id=None)` installs a `PortConnection` on both, and
`connection.transmit(payload, timeline, time=..., priority=..., source=...,
subsystem_id=..., meta=...)` schedules an event at `target_port.owner` carrying a
`PortDelivery`. `Network.wire_ports(wire_id, src, dst, target_action=...)` is the
same thing registered on a network.

A component may own more than one quantum egress port. `EntangledPairSource`
already does (`left_output_port`, `right_output_port` at
`entangled_pair_source.py:207` and `:212`, both via
`sources/_common.py:quantum_output_port`). The delay interferometer's two outputs
need no new mechanism — and `CAPABILITY_MAP.md`'s claim that the DI would be *"the
only component in the repo with two quantum output ports"* is simply wrong.

### 1.3 Sources — `components/sources/`

`SinglePhotonSource` (`single_photon_source.py`, 523 lines) is the template.
Structure worth reusing verbatim, all of it in `sources/_common.py` (770 lines):

| Helper | What it gives the WCP source |
|---|---|
| `ACTION_START` / `ACTION_EMIT`, `EmissionAttempt` | the event vocabulary |
| `schedule_start_event`, `schedule_next_emission_event` | the whole chained one-event scheduler, including the "no event in the past" guard |
| `EmissionTimingProfile` + `DeltaTiming`/`GaussianTiming`/`ExGaussianTiming` | per-slot emission jitter |
| `emission_period_ticks`, `start_time_tick`, `duration_ticks`, `stop_time`, `is_before_stop` | seconds-to-ticks and the active window |
| `validate_source_scalars`, `validate_timing_profile_for_chained_scheduler` | construction-time validation |
| `quantum_output_port` | the egress port |
| `sources/reports.py: store_source_report` | append-locally, transmit-if-connected |

Reusable **as-is**: everything above. That is roughly 90% of `SinglePhotonSource`'s
non-physics body.

Not reusable: `_emit_now`'s payload half — the `StateSampler` draw,
`timeline.qstate.prepare(...)`, the `SubsystemHandle`, and
`apply_noise_models(...)`. A WCP creates no qstate record at all.

One helper needs care. `bind_source_rngs(...)` binds exactly three streams named
`"emission"`, `"state"`, `"timing"` and returns them positionally. The WCP source
has no state to sample and needs two extra streams. It should **not** use this
helper (see 5.3).

`SourcePreparationReport` (`sources/reports.py`) is a frozen dataclass with **no
`__post_init__` and therefore no runtime validation**. It is constructed in exactly
two places, both in `src/simyuj/components/sources/`, and **no code in `src/` or
`examples/` reads `report.state_ref` or `report.state_targets`** — searching for
both returns zero hits outside the definition. Its `state_ref: StateRef` and
`state_targets: tuple[SubsystemId, ...]` annotations are the only obstacle to a
report with no qstate behind it, and mypy runs non-strict
(`pyproject.toml: [tool.mypy] strict = false`).

### 1.4 Channels — `components/channels/quantum.py`

`QuantumChannel` (553 lines) handles one action, `ACTION_TRANSMIT_QUANTUM`,
carrying a `PortDelivery` whose payload is a `Signal`. Per signal it:

1. resolves `qstate_targets_from_signal(signal)` — **line 420, unconditional**
2. samples Bernoulli survival at `10**(-L_total/10)`; on loss, discards the qstate targets and returns
3. samples timing jitter (`_timing_rng.normal`, clamped at zero)
4. calls `timeline.qstate.apply_noise_models(...)` for the flight duration
5. appends `meta`/`timing_meta` via `signal._with_metadata`
6. transmits at `arrival_time` with `self.delivery_priority`

Step 1 is the blocker: `qstate_targets_from_signal` raises
`ValueError("quantum signal must carry state_ref")` for any signal without one.

Two RNG streams, declared in `bind()`: `(channel_id, "quantum_channel", "loss")`
and `(..., "timing")`. Survival probability of exactly 1.0 short-circuits without
consuming the loss stream; zero jitter short-circuits the timing stream. Both
short-circuits matter for the amplitude path.

Propagation is the *scheduled time* of the downstream event — there is no internal
propagation event. Good: adjacent-pulse spacing is preserved exactly, which is what
the interferometer needs.

### 1.5 qstate target resolution — `components/quantum_targets.py`

`qstate_targets_from_signal(signal)` requires a non-`None` `state_ref` and exactly
one `SubsystemHandle`. It has exactly four call sites in `src/`:

| File | Line | On the DPS path? |
|---|---|---|
| `channels/quantum.py` | 420 | **yes** |
| `detectors/detector_array.py` | 395 | yes, if `DetectorArray` is reused |
| `detectors/bell_analyzer.py` | 711 | no |
| `memories/quantum_memory.py` | 1532 | no |

### 1.6 Detectors — `components/detectors/`

Two layers, cleanly separated, and the separation is what makes this cheap.

**`SinglePhotonDetector`** (`single_photon.py`) is *not* a `Component`. It owns
`dead_until` and `last_click_time`, no ports, no timeline, no qstate. Its entry
point is `evaluate_window(*, time, signal_present: bool, window_duration_ticks,
rngs, outcome_label, dark_count_policy, meta)` returning `tuple[RawClick, ...]`,
and `_sample_signal_click` reduces to `rng.random() < self.params.efficiency` when
`signal_present` is true. **`signal_present` is a bool.** That is the one place
this stack assumes a single photon.

`SinglePhotonDetectorParams`: `efficiency`, `dark_count_rate_hz`,
`dead_time_ticks`, `jitter_stddev_ticks`, `p_afterpulse`, `afterpulse_decay_ticks`,
`photon_number_resolving`.

**`DetectorArray`** (`detector_array.py`, 708 lines) is the `Component`. Its
`_handle_detect_signal` (line 386) calls `qstate_targets_from_signal` on line 395
before anything else, then chooses a measurement, executes it against qstate, maps
the result to `DetectorExposure`s through a `ReadoutLayout`, runs
`evaluate_detector_windows`, and resolves clicks.

The measurement layer cannot be neutralised for a signal with no qstate.
`Measure.none()` still reaches `resolve_measurement_targets`, which calls
`_check_unique_targets` — and that raises `ValueError("measurement targets must be
non-empty")` on an empty tuple (`measurement.py:275`). There is no configuration
of `DetectorArray` that accepts an amplitude-only signal.

But the layer *below* it is already payload-agnostic.
`ThresholdClickResolver.resolve` takes `signal: Signal | None`,
`qstate_result: object | None`, **and `measurement_call: MeasurementCall | None`**,
and `_validate_resolve_inputs` (`click.py:112`) explicitly permits `None` for all
three. `DetectionReport` gives every qstate-flavoured field a `None` default.
`evaluate_detector_windows`, `DetectorExposure`, `RawClick`, `bind_detector_rngs`,
and the gate/window helpers touch no qstate at all.

That asymmetry decides section 3: the detector *physics* is reusable unchanged;
only `DetectorArray`'s measurement wrapper is not.

### 1.7 RNG and determinism — `engine/rng_manager.py`, `engine/timeline.py`

`Timeline.rng(*path)` returns a `DeterministicRNG` for a hierarchical string path.
`DeterministicRNG` exposes `random`, `uniform`, `normal`, `poisson`, `exponential`,
`choice`. **`Timeline.rng` raises `RuntimeError` if a new stream is requested after
execution has begun**, so every stream a component might ever draw from must be
declared in `bind()`, including ones only some pulses consume.

`DeterministicRNG.poisson` exists. Under the decided physics it must never be
called on a photon number. It is not needed anywhere in this design.

### 1.8 Validation, logging, tests

`primitives/validation.py` has `require_positive_real`,
`require_non_negative_real`, `require_probability`, `require_finite_real`,
the `require_optional_*` family, `validate_bool`, and friends — all taking a
`field_name` keyword, all raising `TypeError`/`ValueError`. There is **no
complex-number validator**; the new fields need one (section 3, S2).

`tracing/sinks.py:197 _to_jsonable` passes `str`, `int`, `float` and `bool`
through, recurses into tuple/list/dict, and **falls back to `repr(value)` for
everything else**. A `complex` in log meta becomes the string `"(0.447+0j)"` — not
a crash, but not machine-readable either. Discipline, not a code change: **log
derived `mu` and `phase_rad` floats, never the raw `alpha`.**

Test scaffolding: `tests/support/mock_components/` currently holds `ChannelStub`,
`DetectorStub`, `EmitterStub`. The brief's count is exact — `class QuantumSink`
appears in **5** files (`components/memories/test_quantum_memory_e2e.py:72`,
`components/memories/_quantum_memory_support.py:74`,
`components/sources/test_entangled_pair_source.py:31`,
`components/test_components_quantum_channel.py:65`, `network/_components.py:28`)
and `class ReportSink` in **4** (`components/detectors/test_array.py:73`,
`components/detectors/test_bell_analyzer.py:43`,
`components/detectors/test_qubit_readout.py:31`,
`engine/test_end_to_end_topology.py:33`).

### 1.9 How an agent recovers a source's choices — `examples/bb84/agents.py`

The established pattern, worth copying exactly: the source writes its classical
choice into `SourcePreparationReport.sampler_label`; the agent's `on_report`
type-checks the report, decodes the label through a pure helper
(`bb84_basis_bit(report.sampler_label)`), and indexes the result by `signal_id`
and by `attempt_index`. The agent never reads qstate. `AgentReportEndpoint`
(`control/reports.py`) normalizes `PortDelivery`, `AgentReport`, and raw-report
payloads before dispatch.

---

## 2. `Signal`'s final shape

Three new fields, **appended last in this order**, after `validation_flag`. Decided
once, here, for every consumer — source, channel, interferometer, optical detector,
and the step-5 polarizing beamsplitter.

```python
coherent_state: CoherentState | None = None
temporal_mode_sigma_s: float | None = None
polarization: tuple[complex, complex] | None = None
```

**Append, never group by meaning.** These belong conceptually next to
`wavelength_nm`, and they go at the end anyway. Appending makes "is any call site
constructing `Signal` positionally?" *unanswerable* rather than answered once —
including for a call site added later. The reference did the same and left a
comment saying not to tidy it back; keep that comment. (Its AST scan found zero
positional `Signal(...)` constructions anywhere, including inside notebooks, so
this costs nothing today and stays safe tomorrow.)

**`coherent_state`** — a `CoherentState` value object holding one field,
`alpha: complex`. `mean_photon_number` and `phase_rad` are `@property`, derived on
access, never stored. `None` means "this signal carries no optical amplitude" and
is how the channel and detector branch.

This reverses an earlier draft that put a bare `complex` here. Section 11.1 gives
the full reasoning; the short version is that the class centralises validation,
reads better at call sites, and — decisively — `interfere` must be able to return
a state whose μ is *not* `|alpha|²`, which a bare complex cannot express.

**`temporal_mode_sigma_s`** — the **field** envelope's standard deviation in
seconds, for `f(t) = (pi*sigma**2)**-0.25 * exp(-(t-t0)**2 / (2*sigma**2))` with
`integral |f|**2 = 1`. It lives on `Signal` rather than inside `CoherentState`
because an amplitude does not say *when* the light is: σ is a property of the
mode, not of the state occupying it. Two pulses can share a σ and differ in α.
Interferometry needs both.

**σ is not converted with `seconds_to_ticks`.** That helper rounds to integer
picoseconds, which would quantize γ and make a partial-overlap test unwritable at
small Δt. `CLAUDE.md`'s seconds-to-ticks rule governs event *times*, which must be
integers; a continuous width belongs to `duration_s`'s family and stays a float.

**`polarization`** — the Jones vector `u = (u_H, u_V)`, normalized so
`|u_H|**2 + |u_V|**2 == 1`. Also a mode property, for the same reason as σ, which
is why it sits beside it on `Signal` and not inside `CoherentState`. It is a
**descriptor of the occupied mode**, not an independent quantum system: the
physical field is `(alpha*u_H, alpha*u_V)`, one amplitude times a direction. `None`
means the signal is unpolarized-by-omission (DPS and COW) and every polarization
branch is skipped.

Note the resulting split, which is the shape to preserve: **one state field and two
mode fields.** A fourth optical property, if one ever arrives, is a mode property
too and appends after these.

### What is deliberately *not* on `Signal`

**`Theta` and `phi_enc` do not appear.** The split between carrier phase and
encoding phase is load-bearing **at preparation** — a source that randomizes
`Theta` per pulse destroys DPS, because the differential phase picks up
`Theta_n - Theta_{n-1}`, and a source that holds `Theta` across a block leaves
that difference exactly zero. But once `alpha = sqrt(mu) * exp(i*(Theta +
phi_enc))` is built, no downstream device can separate the two, and none needs to:
the interferometer sees only `arg(alpha_n) - arg(alpha_{n-1})`. So the two named
quantities live in the source's configuration and in its preparation report, and
their **sum** lives on the signal. Putting them on `Signal` would be three fields
for two degrees of freedom — one level up from the `alpha`/`mu`/`phase` trap the
brief already rules out.

**`mu` and `phase_rad` do not appear.** Derived, per the same rule.

**No `pulse_index` field.** `SinglePhotonSource` already carries `photon_index`
and `attempt_index` in `meta`, and the interferometer needs to record *two*
contributing indices per output slot. `meta` handles both; a field would handle
neither well.

**No new `SignalKind` or `EncodingScheme` member.** `SignalKind.PULSE` exists and
is unused. `EncodingScheme.PHASE` is the honest label for DPS.

### Identity and timing, unchanged but load-bearing

`id` must stay unique across a run. The interferometer emits *new* signals (like a
memory re-emission, unlike an in-flight transform), so it must namespace them —
`f"{device_id}:slot:{n}:out{p}"` — rather than reuse the input id.

`emission_time` stays the source's tick. Arrival times live in `timing_meta`
under `channel_arrival_time`, which the channel already appends.

---

## 3. The complete shared-code change list

Everything outside a new package that must change. **All of it lands in one
commit, before any component is built**, so every component step afterwards is
purely additive.

Estimated size: roughly 180 lines changed across 8 source files and 1 doc, plus
about 120 lines of new test-support code. Nothing here changes existing
behaviour; the 1460-test baseline must still be 1460 green after this commit
alone.

### S1 — `signal/signal.py`: three fields, appended last

Add the three section-2 fields **after `validation_flag`**, in the order given
there, each defaulting to `None`. Not grouped next to `wavelength_nm` where they
belong conceptually — see section 2 for why appending is the rule. Carry the
"do not tidy these into place" comment across from the reference.

`CoherentState` is new code in `primitives/coherent_state.py` (section 4), so S1
also adds that module, registers it in the package's lazy loader, and adds the
import. It must not live under `components/` -- that is a circular import; see the
placement note in section 4.

### S2 — `signal/signal.py`: validate them in `__post_init__`

Behind the existing `validation_flag` guard, and only when not `None`:

- `coherent_state`: `isinstance(x, CoherentState)`. Nothing more — the value object
  validated its own `alpha` at construction, which is one of the reasons it exists.
- `temporal_mode_sigma_s`: positive finite real. Reuse `require_positive_real`.
- `polarization`: a 2-tuple of complex with
  `abs(|u_H|**2 + |u_V|**2 - 1) <= 1e-12`.

**No addition to `primitives/validation.py`.** An earlier draft proposed
`require_finite_complex` there; with `CoherentState` owning amplitude validation,
the only complex-valued check left is the Jones-vector norm, which is one
expression at one site. `coherent_optics.py` reuses the existing `require_finite_real`,
`require_non_negative_real`, `require_positive_real`, and `require_probability`
exactly as the reference does. `primitives/` is untouched by this commit.

### S3 — `signal/signal.py`: replace the hand-copy with `_derived`

**Do not add three `object.__setattr__` lines to `_with_metadata`.** Replace the
mechanism. `_derived(signal, **replacements)` iterates `_SIGNAL_FIELD_NAMES`, read
once from `dataclasses.fields(Signal)`, and substitutes only what the caller names;
`_with_metadata` stays as a delegating wrapper with an unchanged signature, so its
single existing caller does not move.

Three details are load-bearing, all from the reference (section 11.2):

- **A `_KEEP` sentinel, not `None`.** `None` is a legal value for `coherent_state`;
  clearing an amplitude and preserving one must stay distinguishable.
- **Not a fresh `Signal(...)` at the transform site.** That fails *silently* — a
  field added later gets its declared default at every construction site with
  nothing raising. Strictly worse than today's `AttributeError`.
- **Not a `_with_coherent_state` sibling.** S5's channel mutates amplitude *and*
  metadata, so two siblings means two hand-copies to keep in sync.

Measured there at 1.5× the hand-copy (882 ns → 1287 ns), about 8% of one full
`QuantumChannel._transmit_now` hop. A precomputed `(get, set)` variant recovers most
of it and keeps the `fields(Signal)` derivation; not worth taking until something
profiles signal copying as a bottleneck.

This is what makes every later step additive: after S3, a new `Signal` field is
carried through the channel with no edit anywhere. Ship it with the S10 test that
asserts `_SIGNAL_FIELD_NAMES == fields(Signal)`.

### S4 — `components/quantum_targets.py`: a named predicate

```python
def has_qstate_payload(signal: Signal) -> bool:
    """Whether this signal carries a qstate record the caller must operate on."""
    return signal.state_ref is not None
```

Four call sites of `qstate_targets_from_signal` exist (1.5). Only
`channels/quantum.py` gates on the predicate in this commit; the other three keep
their current unconditional behaviour, which stays correct because nothing routes
an amplitude signal to them. A named predicate rather than an inline
`is not None` so the four sites read alike when the others do need it.

### S5 — `components/channels/quantum.py`: branch on payload

The one substantial behavioural change in the commit. Restructure `_transmit_now`
so the three effects are independent branches over one signal, not a type switch:

```
if has_qstate_payload(signal):      Bernoulli survival; discard on loss;
                                    apply_noise_models; jitter -- exactly as today
if signal.coherent_state is not None: alpha -> sqrt(eta)*alpha, deterministic
                                    optional alpha -> alpha * exp(i * delta_phi)
if signal.polarization is not None: polarization noise   [field now, branch later]
```

New config fields on `QuantumChannel`, both defaulting to `0.0`:
`phase_noise_stddev_rad`, `polarization_rotation_stddev_rad`.

New RNG streams, declared in `bind()` alongside the existing two:
`(channel_id, "quantum_channel", "phase")` and `(..., "polarization")`. They must
be declared even when unused — `Timeline.rng` refuses a new stream after execution
begins (1.7) — and must not be *drawn* when their stddev is zero, matching the
existing loss and timing short-circuits.

**Declaring them cannot perturb existing replays.** `RNGManager.rng` builds each
stream as `SeedSequence(entropy=root.entropy, spawn_key=_path_to_spawn_key(path))`
(`engine/rng_manager.py:338-341`), so a stream's values depend on its *path* and
never on creation order or on how many streams exist. Verified in source; assert it
in a test rather than re-deriving it next time.

**Do not touch the `ready` log.** `test_components_quantum_channel.py:398` asserts
`dict(ready.meta)` by **exact equality** against a five-key dict, so adding
`phase_noise_stddev_rad` there breaks a passing test for no gain. Surface the new
config per-pulse on the forward record instead. This is a deliberate omission, not
an oversight — the reference hit the same wall and made the same call.

Semantics that follow from the branch structure, and that a reader of the counters
must know:

- **`eta` is one physical property with two correct consequences.** The same
  `10**(-L_total/10)` is a survival *probability* for a qstate payload and a power
  *transmission* for an amplitude: `alpha -> sqrt(eta)*alpha`, so `mu` scales as
  `eta`, not as `sqrt(eta)`.
- **Nothing is discarded on the amplitude path.** `lost_count` stays 0 and
  `delivered_count == received_count` however lossy the fiber is. Reading
  `channel_lost == 0` as "lossless" is a trap. Total attenuation delivers coherent
  vacuum; deciding no photon was seen is the detector's job.
- **The loss RNG is never consumed on the amplitude path.** Attenuation is
  arithmetic. An all-amplitude run replays identically regardless of loss config.
- **Metadata is `channel_power_transmission`, not `survival_probability`,** on the
  amplitude path. A pulse that faced no Bernoulli trial must not carry a record
  claiming it did.

Two configurations are **rejected at the moment an amplitude signal arrives**,
not at construction — the channel cannot know at construction which payloads it
will carry:

- `timing_jitter_stddev_ticks > 0`: independent per-pulse jitter destroys the
  adjacent-pulse spacing the interferometer depends on, and can reorder pulses.
- `noise_models` non-empty: Kraus operators are shaped `(2**arity, 2**arity)` and
  have no representation for an optical amplitude.

A signal carrying **both** `state_ref` and `coherent_state` is also rejected. The branch
structure would happily process both, but nothing in this design constructs such a
signal, and accepting one means silently asserting a physical relationship between
a qubit record and a field amplitude that no part of the codebase defines.

New counters alongside `received_count` / `delivered_count` / `lost_count`:
`attenuated_count` for the amplitude branch, and log `mean_photon_number_in` and
`mean_photon_number_out` on the forward record — since loss is invisible in the
counters, it has to be visible in the log.

### S6 — `components/detectors/primitives/readout.py`: one optional field

```python
signal_click_probability: float | None = None
```

on `DetectorExposure`, validated as a probability when not `None`. `None` keeps
today's behaviour exactly.

### S7 — `components/detectors/primitives/window.py`: thread it through

`evaluate_detector_windows` passes `exposure.signal_click_probability` into
`detector.evaluate_window`. Add it to the click `meta` tuple alongside
`readout_signal_present`.

### S8 — `components/detectors/single_photon.py`: honour it

`evaluate_window` gains `signal_click_probability: float | None = None` and passes
it to `_sample_signal_click`:

```python
def _sample_signal_click(self, *, signal_present, signal_click_probability, rng):
    if not signal_present:
        return False
    p = (float(self.params.efficiency) if signal_click_probability is None
         else float(signal_click_probability))
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return rng.random() < p
```

**It overrides `params.efficiency`; it does not multiply it.**
`P = 1 - exp(-eta_d * mu)` already contains `eta_d`. `params.efficiency` still
governs every qstate-backed caller and every dark-count and afterpulse path, which
is why the field is an override rather than a replacement.

> **The trap, recorded so it is not rediscovered.** If `signal_click_probability`
> *multiplies* `params.efficiency` instead of overriding it, `eta_d` is applied
> twice and **every click rate is low by that factor — silently, and
> plausibly.** At `eta_d = 0.2` a run yields a fifth of the clicks it should and
> nothing raises; the rate merely looks like a lossier link, which is exactly the
> quantity a QKD run is trying to measure. The field name invites the wrong
> reading, so state it in the docstring as well as here.
>
> **Required test, not optional.** Assert that a detector with
> `params.efficiency = 0.5` and an exposure carrying
> `signal_click_probability = 1.0` clicks on **every** trial. Under the correct
> override it does; under the multiply bug it clicks about half the time, which no
> other assertion in the suite would catch. Pair it with the converse —
> `efficiency = 1.0`, `signal_click_probability = 0.5` — so neither value can be
> the one being read by accident.

Draw-count note: the efficiency stream is consumed identically in both modes
(exactly one `random()` when `0 < p < 1`), so this does not perturb any existing
run's RNG sequence.

### S9 — `components/sources/reports.py`: a sibling report, no existing type changed

An earlier draft made `SourcePreparationReport.state_ref` and `.state_targets`
optional so a coherent source could reuse it. **Dropped.** The reference's approach
is both smaller and more honest: add a sibling `CoherentPulsePreparationReport`
next to it and widen `store_source_report`'s annotations, leaving
`SourcePreparationReport` byte-identical.

Smaller, because no existing type changes at all. More honest, because a report
carrying `sampler_index` / `sampler_label` for a source that has no sampler is a
lie — and this source has *four* independent per-pulse choices, which would not fit
one sampler slot anyway.

The reference's version records only the single shared `CoherentState`. Ours carries
the four choices as typed fields, each with its alphabet index; see 5.4. No `meta`
namespace, no `wcp.` string keys — the choices are the report's whole purpose and
deserve to be typed.

`store_source_report` stamps `report_kind="source_preparation"`, which is correct
for a source. (The reference's modulator needed a private `_store_report` precisely
because that stamp would have misdescribed it — a good reason that does not apply
here, since the modulator is gone.)

### S10 — `tests/`: two things, both earning their place

**`tests/support/mock_components/signal_sink.py`** — a port-based `SignalSink`,
exported from `mock_components/__init__.py`. The reference already built exactly
this and the name is theirs; keep it. `CLAUDE.md` in this repo also already
documents why the three existing stubs cannot serve: `EmitterStub`, `ChannelStub`,
and `DetectorStub` predate the port layer, own no `Port`, and read
`event.payload_ref` as the payload rather than unwrapping a `PortDelivery`.

Nine inlined copies of `QuantumSink`/`ReportSink` exist today (1.8). This commit
adds a tenth consumer; that is where the duplication stops being tolerable. **Do
not migrate the nine in this commit** — a mechanical change across five test files
has its own review surface, and mixing it with a behavioural commit violates "one
logical change per commit". A follow-up `refactor(tests):`, or nothing.

One correction carried over: a two-port sink is **not** needed for the
interferometer. Two `SignalSink` instances with distinct `device_id`s terminate the
two output ports, because each owns its own `"in"` port. The reference predicted it
would need a two-port variant and then found it did not.

If new tests live in a new folder under `tests/`, that folder needs an
`__init__.py`. `tests/signal/` exists today **without** one, so `from ..support ...`
fails from inside it — confirmed on disk.

**One test that makes S3's failure mode impossible:**

```python
def test_with_metadata_preserves_every_field():
    # enumerate dataclasses.fields(Signal), build a Signal with a distinct
    # non-default value for each, call _with_metadata, assert every field
    # except meta/timing_meta is unchanged.
```

This is the single highest-value test in the plan. It does not test the three new
fields — it tests that *any* future field survives, which is the actual defect
class. Without it, S3 is a landmine for the next person who adds a `Signal` field.

### S11 — `CAPABILITY_MAP.md`: correct 3.1, 3.2, 3.3, 3.3b, 5 — **already done**

This one has been applied ahead of the rest of the commit, because `CLAUDE.md`
sends every session to the map first and leaving it wrong for the duration of the
build was the larger risk. The file is 472 → 333 lines; sections 3.3 and 3.3b were
deleted outright and 3.1, 3.2 and 5 cut back to what `src/` contains. It is an
uncommitted working-tree change on `main` — branch before committing it. The rest
of this section records why, and is retained so the decision is auditable.

The document promised `simyuj.signal.CoherentState`,
`simyuj.components.sources.WeakCoherentPulseSource`,
`simyuj.components.modulators.PhaseModulator`,
`simyuj.components.interferometers.DelayInterferometer`, and
`Signal.temporal_mode_sigma_s`. None exist (section 0). `CLAUDE.md` tells every
future session to read it first.

Two options, and the choice matters:

1. **Delete the sections now, restore them per component.** Honest at every commit.
   Costs a small edit in each later commit.
2. **Rewrite them now to describe this design.** The map is right about the
   destination but wrong at every intermediate commit.

**Recommend (1).** A capability map that describes what does not exist is exactly
the failure this session is meant to prevent, and a map that is right only at the
end is the same failure with a delay. Delete 3.3 (Modulators) and 3.3b
(Interferometers) outright, cut 3.1 and 3.2 back to what ships, and move every
coherent-pulse line in section 5 into a clearly-labelled "planned, not built"
block. The physics content in those sections is good and should be preserved — as
an appendix to *this* document, or in the sections below, which is where most of
it now lives.

### Not changing, and why

- **`DetectorArray`.** See section 6, step 5 — it gets a sibling, not a mode.
- **`components/__init__.py` and `sources/__init__.py` exports.** Each new
  component exports itself in its own commit. Purely additive, no reason to
  front-load.
- **`bell_analyzer.py`, `quantum_memory.py`.** Their
  `qstate_targets_from_signal` calls stay unconditional. Nothing routes an
  amplitude signal to either.
- **`engine/`, `qstate/`.** Nothing at all. This design does not touch the
  timeline, the RNG manager, or the state store. That is the strongest evidence
  that the layer boundaries in `CLAUDE.md` are holding.

---

## 4. Optical arithmetic — module split and surface

**Adopted from `scratch/optical.py`, split across two modules.** That file is 575
lines of finished, well-documented, well-validated code and the arithmetic in it is
correct. Two functions are added for the polarizing beamsplitter and one for
detection; nothing existing is rewritten. Sections 11.1 and 11.3 give the reasoning
for each call.

> **Placement, corrected while building commit 1.** An earlier version of this
> section put everything — the `CoherentState` type *and* the arithmetic — in one
> module under `components/`. **That is a circular import and cannot be built.**
> `Signal` must type-check the value it carries, so `signal/signal.py` imports the
> type; importing it from `simyuj.components.optics` initialises
> `components/__init__.py`, which imports the channels, which import
> `simyuj.signal` while it is still partially initialised. Confirmed empirically:
> `ImportError: cannot import name 'Signal' from partially initialized module
> 'simyuj.signal'`.
>
> The split that works is also the better one, and it is what the brief asked for
> more literally than the single-module version did:
>
> - **`primitives/coherent_state.py` — the definition.** `CoherentState`, its
>   validation, the `mean_photon_number` and `phase_rad` properties, and
>   `from_mean_photon_number`. Nothing else. *Built in commit 1.*
> - **`components/coherent_optics.py` — every operation.** `phase_shifted`,
>   `attenuated`, `split_50_50`, `interfere`, `gaussian_temporal_overlap`,
>   `click_probability`, `polarization_weights`, `rotated_polarization`, each
>   taking and returning `CoherentState`. *Built in step 1.*
>
> "`Signal` carries definitions only — no math, no physics on it" is then true by
> construction, and `coherent_optics.py` genuinely is *one* shared module for all
> optical arithmetic. The reference put both halves under `signal/`, which is why
> it never hit the cycle.
>
> **Why `primitives/` and not `signal/`.** The deciding precedent is
> `SubsystemHandle`: a value type carried by a `Signal` field, referenced across
> component code, living in `primitives/subsystems.py` and deliberately *not*
> re-exported from `simyuj.signal`. `CoherentState` is the same kind of thing in
> the same relationship, and putting it under `signal/` would create a second
> convention for an identical problem. `primitives/__init__.py` loads its
> submodules lazily, so importing `primitives.coherent_state` never pulls in
> `primitives.messages`, which is the one part of that package that does depend on
> `simyuj.signal`.
>
> **Open for step 1:** whether `phase_shifted` and `attenuated` are free functions
> in `components/coherent_optics.py` or stay methods on `CoherentState`. Methods read
> better
> (`state.attenuated(0.1)`); free functions keep all arithmetic in one module.
> Commit 1 does not need either, so the decision is deferred to where the
> arithmetic actually lands. The `attenuated`-not-`with_attenuation` naming
> rationale survives both ways: the parameter name carries the convention.

`components/coherent_optics.py` is imported by the source, the channel, the
interferometer, and the optical detector. No `Signal`, no `Timeline`, no RNG, no
component — so every function is unit-testable without a timeline, which is where
the `|gamma| = 0.6` test lives.

### `CoherentState` — carried over unchanged

```python
@dataclass(frozen=True, slots=True)
class CoherentState:
    alpha: complex                              # the only stored field

    @property
    def mean_photon_number(self) -> float: ...  # re*re + im*im, not abs()**2
    @property
    def phase_rad(self) -> float: ...           # cmath.phase; 0.0 for vacuum

    def with_phase_shift(self, phase_rad: float) -> CoherentState: ...
    def attenuated(self, power_transmission: float) -> CoherentState: ...

    @classmethod
    def from_mean_photon_number(cls, mu: float, *, phase_rad: float = 0.0) -> CoherentState: ...
```

Behaviour worth restating because it is easy to reimplement wrongly:

- **No upper bound on μ.** "Weak" names the source, it is not a constraint. μ = 4 is
  as valid as 0.1.
- **μ = 0 is the coherent vacuum**, a real optical state, not an absent pulse.
  `phase_rad` returns `0.0` for it rather than raising, matching `cmath.phase(0j)`.
- `bool` rejected; non-finite real or imaginary part rejected.
- `attenuated` takes a **power** transmission, so α scales as √η and μ as η.
  The asymmetry with `with_phase_shift` is deliberate: `with_attenuation(0.1)` does
  not say whether 0.1 is the loss or the survivor.
- `with_phase_shift` and `attenuated` **commute analytically, not bit-exactly**.
  Compare with a tolerance.
- `theta = pi` does **not** give exactly `-alpha`: `exp(1j*pi)` is
  `(-1+1.2246e-16j)`. Not special-cased, on purpose — a hidden branch for π would
  make exactness look guaranteed and the interferometer's dark port would rest on an
  implementation accident instead of its own tolerance.

### Module functions

```python
def split_50_50(state: CoherentState) -> tuple[CoherentState, CoherentState]:
    """Both outputs are alpha/sqrt(2). With vacuum on the second input the two
    are equal, so one immutable value object is returned twice."""

def gaussian_temporal_overlap(*, sigma_a_s: float, sigma_b_s: float,
                              delta_s: float) -> float:
    """gamma = sqrt(2*sa*sb / (sa^2+sb^2)) * exp(-dt^2 / (2*(sa^2+sb^2)))

    Field-envelope sigma, so equal widths give exp(-dt^2 / (4*sigma^2)) -- the
    denominator is 4. An intensity-envelope sigma gives 8; the two must never be
    mixed. Keyword-only, because three floats are trivially transposable.
    Zero width is rejected rather than special-cased: the sigma -> 0 limit is a
    different (discrete) model, and a caller who wants it should pass a small
    width."""

def interfere(short_arm: CoherentState, long_arm: CoherentState, *,
              overlap: complex = 1.0) -> tuple[CoherentState, CoherentState]:
    """Recombine two arms at BS2. Port 0 is destructive when the arms are in phase.

    Implemented by splitting the long arm into a component along the short arm's
    mode (weight gamma) and an orthogonal remainder that cannot interfere:

        mixed    = gamma * long.alpha
        amp_k    = (short.alpha -/+ mixed) / sqrt(2)
        residual = max(0.0, 1 - abs(gamma)**2) * long.mean_photon_number / 2
        mu_k     = abs(amp_k)**2 + residual

    which is algebraically identical to
    mu_k = 1/2 [ |a_s|^2 + |a_l|^2 -/+ 2 Re(conj(a_s) a_l gamma) ]
    but keeps the non-interfering light visible in the code.

    Energy is conserved for every input: mu_0 + mu_1 == mu_s + mu_l at any gamma.
    That identity is what catches a convention error, so assert it on every case.

    Vacuum inputs, gamma = 0, unequal amplitudes, the first pulse and the last
    pulse are values of this equation, not branches around it.

    Accepts a complex overlap even though gaussian_temporal_overlap returns a
    real: a relative carrier-frequency offset would make it complex, and allowing
    it now costs nothing."""
```

**`interfere` returns states, and μ does not round-trip bit-exactly.** Because the
residual is added *after* the amplitude is formed, `amp_k` has the wrong modulus, so
the result is rebuilt with
`CoherentState.from_mean_photon_number(mu_k, phase_rad=cmath.phase(amp_k))` — μ
through `sqrt` and back. A μ of 0.2 returns as 0.19999999999999998 and an
analytically dark port lands near 1e-16, or ~1e-33 downstream of the `exp(1j*pi)`
residue. An earlier draft of this section claimed the squared moduli were exact;
they are not. Compare with a tolerance everywhere.

**Intensity-exact, mode-truncated.** At `|gamma| < 1` the field leaving a port is
`a_s f_s(t) -/+ gamma a_l f_l(t)`, a superposition of two non-identical envelopes
that no single `(alpha, sigma)` pair describes. The returned state carries the exact
μ *including the orthogonal residual*, the phase of the interfering component, and
(at the component level) the short arm's σ. **Neither the phase nor the width of an
interferometer output may feed a further phase-sensitive or temporal-mode
interference.** At `|gamma| = 1` all three are exact.

### Added here, not in the reference

```python
def click_probability(mu: float, detector_efficiency: float) -> float:
    """1 - exp(-eta_d * mu).

    Where photon-number stochasticity enters -- in closed form, exactly once.
    Nothing in this codebase samples n from Poisson(mu); the Poisson statistics
    are integrated analytically here. The value already contains eta_d, so a
    caller passing it to DetectorExposure.signal_click_probability must not also
    apply params.efficiency (S8)."""

def polarization_weights(
    polarization: tuple[complex, complex] | None,
) -> tuple[float, float]:
    """(w_H, w_V), w_H + w_V == 1, for splitting mu at a polarizing element.

    For a Jones vector u this is (|u_H|^2, |u_V|^2), which is exactly
    (Tr(Pi_H rho), Tr(Pi_V rho)) for the pure state rho = |u><u|. Every polarizing
    component calls this rather than reading u directly, so replacing the pure
    descriptor with a density matrix later changes this function and nothing else.
    Rejects None -- a component needing a polarization split must not invent one."""

def rotated_polarization(polarization: tuple[complex, complex],
                         theta_rad: float) -> tuple[complex, complex]:
    """Rotate the Jones vector about the propagation axis. Unitary, so the result
    stays pure: this is drift, not depolarization -- see section 9.1."""
```

**Still absent, deliberately:** any sampler. `coherent_optics.py` takes no RNG and returns no
random value, which makes the "never sample n" rule structural rather than a comment.

### Beamsplitter convention

The real 50:50 matrix `(1/sqrt(2)) [[1, -1], [1, 1]]` at both splitters, stated once
at the top of the module and used everywhere including the tests.

The symmetric convention `(1/sqrt(2)) [[1, i], [i, 1]]` is the *same physical
device*; the two give an identical `out_0` and an `out_1` differing only by an
unobservable global `i`. They differ in where the interference term lands — `Im`
for the symmetric one, `Re` for the real one — and `Re` is chosen to match the μ
equations the tests are written against.

Worth knowing, from the reference: the original brief for that work specified BS1 as
`a_l = i*alpha/sqrt(2)` *and* μ equations in `Re(...)`, which are inconsistent as
written. The `i` does not eliminate interference; it moves it to `Im`. If a future
spec mixes the two again, that is the resolution.
## 5. The source's preparation model

One component: `WeakCoherentPulseSource` in
`components/sources/weak_coherent_pulse_source.py`. **The modulator is inside it.**
It is Alice's complete preparation device — it chooses `mu`, `Theta`, `phi_enc`,
and polarization, and constructs the `CoherentState` at emission.

**The reference built this component and its emit path is adopted essentially
verbatim.** `scratch/weak_coherent_pulse_source.py` reuses `_common.py`'s chained
scheduler unchanged, keeps a single `pulse_index` counter with no
`attempt_index`/`emission_index` split (every active slot emits, so there is
nothing to distinguish), sets `state_ref=None`, uses `validation_flag=False` on the
trusted hot path, and ends with the standard log / report / transmit tail. All of
that is right and none of it needs rethinking.

**One thing in it is wrong for our purposes and it is the reason this section
exists.** That source takes a scalar `mean_photon_number`, builds one
`CoherentState` in `__post_init__` (`:227-229`), and attaches *that exact object* to
every pulse — its tests assert identity with `is`. As a guarantee that the amplitude
never drifts, that is elegant. But it makes the component a fixed-amplitude pulse
train generator: no per-pulse μ, no per-pulse phase, no polarization, and exactly
one RNG stream. Every per-pulse decision had to move downstream into a separate
`PhaseModulator`, and decoy states and BB84 polarization would each need another
one. The diff that fixes it is small and local — replace the instance attribute with
four policy draws and build the state inside `_emit_now` — and it is what removes
the need for the modulator component entirely.

A separate modulator would otherwise mean a second event hop, a second signal
identity question, and a `PhaseModulationReport` describing a phase the source
already knows. Its only physical justification would be modelling modulator
insertion loss or finite extinction ratio as distinct from the source, and neither
is modelled anywhere.

### 5.1 Per-pulse choices

Every active slot emits exactly one pulse. There is no emission Bernoulli — a
laser fires; whether a photon is *seen* is the detector's problem. (`mu = 0` is
how you express an empty slot, which is what COW's sequences need.)

```
mu_n       <- intensity selector
Theta_n    <- carrier-phase selector
phi_enc_n  <- encoding-phase selector
u_n        <- polarization selector   (None for DPS and COW)

state_n = CoherentState.from_mean_photon_number(mu_n, phase_rad=Theta_n + phi_enc_n)
```

**The selector pattern is the reference's, generalised from one axis to four.**
`scratch/phase_modulator.py` got the shape right and it transfers directly:

- Each selector is a **frozen** strategy object satisfying a `Protocol`.
- `select_*(index, rng)` is **pure** — it receives the source's pulse counter
  rather than holding a cursor, so one selector instance can drive several
  components without shared hidden state.
- It returns a small frozen record carrying **both the value and its position in
  the alphabet** — `PhaseSelection(phase_rad, index)` and siblings — because the
  index is what the protocol layer decodes and what the report must record. An
  earlier draft here had `choose(rng)` returning a bare value; that loses the index.
- No classical label. `StateSample` needs one because the prepared quantum state is
  opaque; a phase is fully described by its value and located by its index, so a
  label would only restate them. Step 6 decodes the index directly.
- The stated precedent is **`GateModel` in `detectors/primitives/gate.py`** — a
  Protocol with a trivial implementation (`AlwaysOpenGate`), a parametric one
  (`PeriodicGate`), and an explicit-sequence one (`ScheduledGate`) — not
  `EmissionTimingProfile`. `ScheduledGate` also disproves an earlier claim in this
  project that no component takes a pre-drawn driving sequence.
- `min(int(rng.random() * n), n - 1)` for a uniform draw, guarding the
  `random() == 1.0` boundary.
- A sequence selector must **raise on exhaustion** rather than wrapping silently
  unless `repeat=True`: a pattern shorter than the run would otherwise produce a
  plausible-looking but wrong key.

The seams matter more than the implementations:

| Policy | V1 builds | Fits later without changing anything else |
|---|---|---|
| intensity | `FixedIntensity(mu)` | `DecoyIntensities(levels, probabilities)`; a COW sequence policy |
| carrier phase | `FixedCarrierPhase(0.0)`, `PerPulseRandomCarrierPhase()` | `BlockRandomCarrierPhase(block_length)`; a Wiener drift for finite laser linewidth |
| encoding phase | `RandomPhaseChoice(phases=DPS_PHASES)` | `PhaseSequence(...)`, `FixedPhase(...)`; a 4-phase set for DQPS |
| polarization | nothing — always `None` | `RandomJonesChoice(states=BB84_JONES)` for decoy BB84 |

`DPS_PHASES == (0.0, pi)`, `FixedPhase`, `RandomPhaseChoice`, and `PhaseSequence`
are lifted from `scratch/phase_modulator.py:47-222` unchanged apart from moving
into the source module. `FixedPhase` is kept as its own type rather than expressed
as `PhaseSequence((x,), repeat=True)` for the same reason `DeltaTiming` and
`AlwaysOpenGate` exist: "no variation" is worth stating directly in a config.

That is four selectors plus two carrier-phase variants for V1. Every extension in
the right-hand column is a new class in the same file and no change anywhere else.

### 5.2 `StateSampler` is not reusable; the sibling is trivial

`StateSampler` has the right *shape* — a finite classical distribution, an
`index`, a `label`, an explicit caller-supplied RNG — and every one of these
protocols does make a finite discrete choice per pulse (DPS: `{0, pi}`; decoy:
`{signal, decoy, vacuum}`; COW: `{empty, one, decoy}`).

But its payloads are qstate initializers, built at construction through
representation handlers. Measured against the real class, not assumed:

```
StateSampler(states=((0.2, 0.0), (0.2, 3.14159)), probabilities=(.5,.5), rep="ket")
  -> InvalidStateError: ket vector must be normalized
StateSampler(states=("signal", "decoy"),          probabilities=(.5,.5), rep="ket")
  -> InvalidStateError: unsupported ket state: 'signal'
```

There is no `rep` that accepts a `(mu, phase)` tuple, and adding one would put
optical preparation choices inside the qstate representation layer — a layer
violation for a class whose entire job is qstate payloads.

**Recommendation: no sibling in `qstate/`, and no generic sampler at all.** The
four strategy objects in 5.1 *are* the sampler, one per quantity, each about
twelve lines. A generic `PreparationSampler` over frozen `PulsePreparation`
records is the tempting alternative; it is more machinery than four small classes,
and it forces the four independent choices into one joint distribution, which is
wrong — decoy BB84 chooses intensity and polarization independently, and a joint
sampler would have to enumerate their product.

RNG discipline: each policy draws from its own named stream, so adding decoy
levels later does not shift the encoding-phase draw sequence and break replay of
an existing DPS run.

### 5.3 RNG streams

The WCP source binds **four** streams directly rather than using
`bind_source_rngs`:

```
(device_id, "wcp_source", "timing")      emission jitter, via the timing profile
(device_id, "wcp_source", "intensity")   mu
(device_id, "wcp_source", "carrier")     Theta
(device_id, "wcp_source", "encoding")    phi_enc  (and polarization, when added)
```

`bind_source_rngs` binds exactly `"emission"`, `"state"`, `"timing"` and returns
them positionally. The WCP has no emission Bernoulli and no qstate, so two of the
three would be declared and never drawn. A declared-but-never-drawn stream is a
lie in the binding log, and the helper cannot be extended without changing the
signature that `SinglePhotonSource` and `EntangledPairSource` both depend on. Four
explicit `timeline.rng(...)` lines are clearer than a helper that fits badly. The
reference reached the same conclusion for the same reason and bound its one stream
directly, with a comment saying so.

All four must be declared in `bind()` even when a selector is deterministic
(`FixedIntensity` draws nothing), because `Timeline.rng` refuses new streams once
execution begins (1.7). Declaring them is free of replay risk: stream values derive
from the stream path, never from creation order or count (S5, verified at
`engine/rng_manager.py:338-341`).

**A deterministic configuration should consume zero randomness, and that is
testable.** With `DeltaTiming` plus `FixedIntensity`, `FixedCarrierPhase`, and
`PhaseSequence`, the source draws nothing at all. The reference asserted exactly
this with a tripwire RNG that raises if drawn from; adopt that test.

### 5.4 How the agent recovers Alice's choices

Following the BB84 pattern (1.9), but through a **purpose-built report** rather
than `SourcePreparationReport` — see S9. The source emits a
`CoherentPulsePreparationReport` on its classical `report` port via the existing
`store_source_report`; Alice's `NodeAgent` records it in `on_report`, keyed by
`signal_id` and `pulse_index`.

```python
@dataclass(frozen=True, slots=True)
class CoherentPulsePreparationReport:
    report_id: str
    device_id: str
    time: int
    pulse_index: int
    signal_ids: tuple[str, ...]
    coherent_state: CoherentState        # mu and phase derived, not stored twice
    emission_slot_tick: int
    emission_delay_ticks: int
    # the four preparation choices, each with its alphabet index
    mean_photon_number: float
    intensity_index: int
    carrier_phase_rad: float
    encoding_phase_rad: float
    encoding_phase_index: int
    polarization: tuple[complex, complex] | None = None
    polarization_index: int | None = None
```

The first eight fields are the reference's record unchanged. The rest are the per-
pulse choices it had no need for. **No `state_ref`, no `state_targets`, no
`sampler_*`** — this source creates no quantum state and has no sampler, and a
report claiming otherwise would be false in the record step 6 reads.

Typed fields rather than a `wcp.` namespace in `meta`, reversing an earlier draft:
the choices are this report's entire purpose, and four of them do not belong in an
untyped bag when they will be read on every pulse of every run.

`Theta` and `phi_enc` are recorded **separately**, not pre-summed. Their sum is
already in `coherent_state.phase_rad`; keeping them apart is what lets a later
analysis attribute a visibility loss to carrier drift versus encoding.

**The applied phase is not recoverable from the amplitude, and this is the trap the
reference documented most insistently.** `CoherentState.phase_rad` is
`cmath.phase(alpha)` — the *total wrapped* phase. After the carrier and encoding
phases sum it already differs from either, and after S5's channel phase noise it
differs more. Alice's agent must reconstruct the differential phase from the
reported `encoding_phase_index`, never from `arg(alpha)`. Doing otherwise is the
"protocol knowledge is earned" violation in `CLAUDE.md` invariant 6: reading a
physical quantity no message conveyed.

Alice's decode helper is pure and lives in `helpers.py`:
`dps_differential_bit(index_prev, index_curr) -> int` — indices, not radians, so a
floating-point phase never enters a bit decision. `0` is phase `0` and `1` is
phase `pi`.

### 5.5 What V1 builds versus what merely has to fit

**Builds:** the source component, `FixedIntensity`, `FixedCarrierPhase`,
`PerPulseRandomCarrierPhase`, `RandomPhaseChoice`. `polarization` is always `None`.

**Fits, not built:** decoy intensity levels, block carrier phase, Jones-vector
polarization choice, DQPS 4-phase encoding, COW sequence policies. Each is one new
class in `weak_coherent_pulse_source.py`.

---

## 6. Build order

Each step is one commit, on a branch, with counters checked before moving on
(`CLAUDE.md`: "Build in this order, one stage at a time"). Step 0 is the only
commit that touches shared code; every later step is additive.

### Step 0 — the shared commit: `refactor(components): admit coherent-amplitude signals`

All of section 3 (S1 through S11). No new component. Baseline must still read
1460 passed, plus the `_with_metadata` completeness test.

The commit body must state the determinism implication explicitly — two new RNG
streams on `QuantumChannel`, never drawn at their zero defaults. Per `CLAUDE.md`,
a commit that adds an RNG stream name is not a `chore`.

### Step 1 — `components/coherent_optics.py`. Depends on: nothing

Pure functions, no `Signal` import. Testable and tested alone.

Tests that earn their place: the `interfere` energy invariant
`mu_0 + mu_1 == mu_s + mu_l`, swept over gamma and over unequal amplitudes; the
20:80 split at `abs(gamma) = 0.6` with equal amplitudes and zero differential
phase; `gaussian_temporal_overlap` reducing to `exp(-dt**2 / (4*sigma**2))` at
equal widths — the test that pins the 4 and rejects the 8; `attenuated` scaling
`mu` by `eta` and not by `sqrt(eta)`; `phase_shifted` preserving `abs(alpha)`
exactly.

Not worth a test: `mean_photon_number` and `phase_rad`, which are `abs()**2` and
`cmath.phase`.

### Step 2 — `WeakCoherentPulseSource`. Depends on: S1, S2, S9, optics.py

Reuses the entire `sources/_common.py` scheduler. Exports itself from
`sources/__init__.py` and `components/__init__.py`.

Counter check: `prepared_pulses == active_slots` exactly (no emission Bernoulli),
every report has `state_ref is None`, and `abs(alpha)**2 == mu` for every pulse.

Determinism check before moving on: two runs at the same `master_seed` produce
identical `phi_enc` sequences; different seeds do not.

### Step 3 — the channel's amplitude path. Depends on: S5

Landed in step 0, exercised here: source, channel, `QuantumSink`.

Counter check: `channel_lost == 0` at every attenuation; `mu_out == eta * mu_in`;
adjacent-pulse arrival spacing exactly equal to the emission period when the
timing profile is `DeltaTiming`. The last of these is the precondition the
interferometer depends on, and it is far cheaper to assert here than to debug
there.

### Step 4 — `DelayInterferometer`. Depends on: S1, S2, S3, optics.py

Two quantum egress ports (`out_0`, `out_1`), both of which must be connected.
Holds one long-arm contribution at a time; self-schedules `ACTION_RESOLVE_BS2` and
`ACTION_FLUSH_DELAY_ARM`. `BellStateAnalyzer` is the precedent to copy — it already
buffers port arrivals and self-schedules `ACTION_COINCIDENCE_TIMEOUT` for the
unmatched case (`bell_analyzer.py:640`, `:838`), which is structurally the same
problem.

Declares **no RNG streams** — the device is ideal by specification, and declaring
a stream that is never consumed would be the same lie as in 5.3.

**Three structural decisions, all adopted from `scratch/delay_interferometer.py`
(1023 lines, complete). Do not re-derive them:**

- **`vacuum_like(other)` removes every edge branch** (`:247-273`). It returns a
  vacuum contribution borrowing the present arm's BS2 tick, σ, and wavelength, with
  `pulse_index=None`. Because the interference term is proportional to *both*
  amplitudes, a vacuum partner kills it at any γ — so the first pulse, the last
  pulse, and the flush all go through the identical `interfere()` call. It also
  collapses the output-σ rule to one line: the output always carries the short
  arm's σ, which via `vacuum_like` is the long arm's on a flush.
- **Combine on arrival, holding the previous long arm** (`_handle_arrival`,
  `:628-682`). Take the holder first (`held, self._held = self._held, None`),
  combine or defer, then refill with this pulse's own long arm. *Not* one scheduled
  BS2 event per arm: at the design point τ = T, pulse *k*'s short arm and pulse
  *k−1*'s long arm land on the **same tick**, and any two-event scheme makes
  correctness depend on same-tick ordering.
- **The causality deferral.** Combine-on-arrival is acausal when a pulse arrives
  *before* the held arm has reached BS2 — reachable whenever the source uses a
  stochastic timing profile, which shortens spacing below τ. Guard with
  `if partner.bs2_tick <= arrival_tick: resolve else: defer`, deferring to
  `ACTION_RESOLVE_BS2` at the long arm's BS2 tick and releasing the holder in the
  same step. Do **not** reject early arrivals instead: that turns a legitimate
  timing profile into a seed-dependent mid-run abort, the worst failure shape
  available in a deterministic simulator.

`BellStateAnalyzer` is the structural precedent for the holder plus self-scheduled
deadline plus stale-check pattern (`bell_analyzer.py:453`, `_schedule_timeout`,
`_pop_buffer_by_id`), but narrower here — one holder, no pairing key, no FIFO
fallback — because each arrival consumes and refills it within one event.

Consequences to check in the counters, all of them real and all of them surprising
the first time:

- **An N-pulse train gives N+1 output slots.** The first pulse's short arm and the
  last pulse's long arm each meet vacuum, split 50:50, and carry no bit. Energy
  ledger: `½μ + (N−1)μ + ½μ = Nμ`.
- **Nearest-neighbour pairing only.** `tau` approximately equal to the pulse period
  is the supported regime. τ = 2T is a section-9 gap, not a configuration.
- A held arm is flushed against vacuum at `arrival + 2*tau`. **The deadline is that
  nearest-neighbour assumption, not a decay estimate** — the "γ ≈ 0 at Δt = τ"
  justification only holds for σ ≪ τ; at σ = τ the discarded overlap is ~0.78.
  Assert that number in a test so the assumption is visible in the suite. A pulse
  arriving at exactly `arrival + 2*tau` is processed before the flush and **does**
  pair, at Δt = τ; one tick later it does not. Energy is conserved either way.
- `flush_priority` (10000) must stay **strictly** above the upstream
  `delivery_priority` (0). Equality is *worse* than inversion: the tie falls to
  `event_id`, so the outcome depends on scheduling order. Pin all three cases,
  including equality, because the coupling is to a value configured on a
  *different* component.
- A run must reach `last_arrival + 2*tau` — and any outstanding deferred
  resolution tick — or the final slots never execute. This bites whoever writes the
  first end-to-end trial.
- Outputs are new signals with new ids (section 2). This is the "transform in
  place → derive; new optical event → construct" rule: the interferometer
  constructs fresh `Signal`s rather than going through `_derived`, following
  `QuantumMemory._make_emitted_signal`.

**Do not copy the per-method ten-key log block.** It is justified in the
interferometer for a reason specific to it — the device deliberately does not
validate τ against the pulse period, so a τ/T mismatch is visible *only* in the run
record, which is why `temporal_overlap`, `delta_ticks`, both BS2 ticks, and μ
in/out belong in the `interfered` record. A component that validates its
configuration up front has no such obligation and should log far less.

Size expectation, from the reference: 544 code lines, of which the physics inside
`_resolve` is **7** — a tick delta, one `gaussian_temporal_overlap` call, one
`interfere` call. The rest is three actions, three rejections, four ports, two
output ports to emit on, five frozen records, and roughly a third of all method
bodies being logging and reporting. That last proportion matches
`PhaseModulator._modulate_now`, so it is the house convention, not this component.

### Step 5 — `OpticalDetectorArray`. Depends on: S6, S7, S8, optics.py

**A sibling of `DetectorArray`, not a mode of it.** The reasoning, since this is
the one place the design deviates from the obvious move:

`DetectorArray`'s spine is *measure the qstate, map the outcome to detectors*. For
an amplitude pulse there is no measurement — and it cannot be switched off:
`Measure.none()` still reaches `_check_unique_targets`, which raises on an empty
target tuple (1.6). Making `DetectorArray` accept an amplitude signal means
conditionally disabling `measurement`, `readout`, and `consume_signal` on a
708-line component covered by a large share of the 1460 tests.

The layer below it is already payload-agnostic, and that is where all the physics
is. `OpticalDetectorArray` reuses **unchanged**: `SinglePhotonDetector`,
`SinglePhotonDetectorParams`, `evaluate_detector_windows`, `DetectorExposure`,
`RawClick`, `bind_detector_rngs`, the gate and window helpers,
`ThresholdClickResolver` (whose `resolve` already accepts `signal=None`,
`qstate_result=None`, and `measurement_call=None`), and `DetectionReport`. It
writes only the roughly 80 lines that are genuinely different: one ingress port
per detector, and

```
mu_j       = signal.coherent_state.mean_photon_number
exposure_j = DetectorExposure(
                 detector_id=...,
                 signal_present=True,
                 signal_click_probability=click_probability(mu_j, eta_j))
```

Two components then share one detector-physics core, and the shared-code cost is
the three-file `signal_click_probability` thread (S6 through S8) instead of a
second mode in a component that has no room for one.

**Hand-off constraints from step 4, recorded by the reference so step 5 does not
rediscover them:**

- Each output port delivers the **exact** μ for that port. Sampling a click from it
  is step 5's job; the interferometer never decides.
- **A perfect dark port delivers μ ≈ 1e-33, not 0** — the `exp(1j*pi)` residue,
  squared. `CoherentState(0j)` with exactly μ = 0 is *also* deliverable, from total
  upstream attenuation. The detector must produce "no click" from its own
  statistics in both cases; `1 - exp(-eta*mu)` does this correctly with no special
  case, which is one more reason the closed form beats a branch.
- The final flush lands at `last_arrival + 2*tau`, up to two slot periods after the
  last real arrival. A **gated** detector must still be open then, or accept losing
  the last slot.
- The two edge slots split 50:50 and carry no bit. **Step 6 discards them; step 5
  must not special-case them** — they are ordinary pulses to a detector.
- Join on `short_pulse_index` / `long_pulse_index` in the signal meta, never on an
  interference counter. `None` means that arm was vacuum.
- `temporal_overlap` travels in the signal meta and the report; step 6 wants it for
  a visibility budget.

### Step 6 — agents, trial, reporting

`configs.py`, `helpers.py`, `agents.py`, `trial.py`, `reporting.py`, `demo.py`
under `examples/dps/`, per `CLAUDE.md`'s layout. Alice records preparations from
`CoherentPulsePreparationReport`; Bob records `DetectionReport`s; Bob announces detection
*times* on the classical channel; Alice maps each announced slot to
`phi_n - phi_{n-1}`. Sifting is by detection time, not by basis — there is no
factor-of-two basis-matching loss in DPS.

The agent must drop the two boundary slots from step 4's N+1. Slot 0's short arm
met vacuum, so both detectors fire with equal probability and Alice has no
`phi_-1` to pair with. It is 2 slots out of N and vanishing in a long run, but it
must be an explicit drop with a counter, not an incidental one.

### Step 7 — the polarizing beamsplitter and decoy BB84

Section 8. Not now.

### Dependency summary

| Step | Needs from the shared commit |
|---|---|
| 1 optics.py | nothing |
| 2 source | S1 `coherent_state`, S2 validation, S9 sibling report |
| 3 channel path | S1, S3 `_with_metadata`, S5 branch |
| 4 interferometer | S1 `temporal_mode_sigma_s`, S3 |
| 5 optical detector | S6, S7, S8 |
| 6 agents | nothing new |
| 7 PBS / decoy | S1 `polarization`, plus the deferred S5 polarization branch |

---

## 7. The two open questions

### 7.1 Can the qstate store hold one polarization record per pulse at GHz over a long run?

**No, by roughly two orders of magnitude — and it is a throughput problem, not a
memory problem.** Measured on this machine against the real `QuantumStateManager`
(throwaway script in the session scratchpad; nothing written to the repo):

| Path | Per pulse | Extrapolated to 1e9 pulses (1 s at 1 GHz) |
|---|---|---|
| `prepare("\|0>")` + `discard(...)`, steady state | **14.66 us** | **about 4.1 hours**, qstate work alone |
| `prepare` only, never discarded | 36.44 us | 10 hours, and 675 GB retained |
| `alpha` complex arithmetic (`sqrt`, `exp`, `*`, `abs`) | **0.29 us** | about 5 minutes |
| `Signal(..., validation_flag=False)` construction | 1.16 us | about 19 minutes |

The store itself is well-behaved: `discard` deletes the record when every
subsystem goes, `_records` returns to size 0 after a churn loop, and `_next_ref`
is a plain int. **Retained memory is bounded by live records, not cumulative
ones** — so a discipline of discarding each pulse keeps the footprint flat. The
cost is per-operation, not per-run: about 14.7 us of layout construction,
validation, and two dict mutations, against 0.29 us for the arithmetic that would
replace it. That is a **~50x** ratio, before any of the event scheduling, channel,
and detector work that dominates the rest of the loop.

So the answer is not "qstate is slow". It is that a per-pulse qstate record buys
nothing here. Polarization on a WCP is a **mode descriptor**, not an independent
quantum system: two pulses in the same polarization mode are not two entangled or
even two distinguishable systems, they are two excitations of one mode. Storing a
qubit per pulse would model a physical claim that is false.

**The alternative is the one already decided: the Jones vector on `Signal`,** with
`polarization_weights()` in `coherent_optics.py` as the single accessor (section 4). Its
limits, stated plainly:

- A Jones vector is pure. `rotated_polarization` (drift) is unitary and stays
  pure; **depolarization has no representation** (9.1).
- Where the qstate path *is* still viable: runs under about 1e6 pulses, which
  covers every tutorial and most tests. Nothing in this design forbids a
  qstate-backed polarization source; it just is not this source.

### 7.2 Does `Theta` need block structure in V1?

**No — per-pulse versus fixed is enough, provided `Theta` is chosen through a
strategy object rather than a bare float.**

The seam is the thing that must exist up front; the third strategy is ten lines
added later with no other change. Concretely:

| Theta policy | Contribution to the differential phase | DPS visibility |
|---|---|---|
| fixed for the run | `Theta_n - Theta_{n-1} == 0` exactly | 1 |
| per-pulse uniform | uniform on `(-pi, pi]` | 0 |
| block of length L | 0 within a block, uniform at one boundary per block | `1 - 1/L` |

The two extremes **bracket** the physics: any block model must land between them,
and they are the pair that makes a test meaningful. Two tests — visibility 1 under
`FixedCarrierPhase`, visibility approximately 0 under
`PerPulseRandomCarrierPhase` — pin the sign and the mechanism. A block test at
L = 100 asserting `~0.99` adds a number, not a mechanism.

Two things worth knowing before choosing the default:

**The protocols disagree about what `Theta` should do, and that is exactly why it
is a policy.** DPS and COW need `Theta` coherent across the train — that coherence
*is* the encoding. Decoy-state BB84 wants `Theta` randomized **per pulse**, because
per-pulse phase randomization is what makes the emitted state a Poisson mixture of
Fock states and the decoy-state analysis valid. A source shared across all three
cannot have one right answer here, and cannot express the disagreement as a scalar.

**`FixedCarrierPhase` means infinite laser coherence length.** That is an
approximation, and it is optimistic. The honest model of finite linewidth is a
Wiener process on `Theta` — which is the *same* structure the channel's phase noise
needs (9.3), and the same seam. Recommend a V1 default of `FixedCarrierPhase(0.0)`
and a line in the run report saying laser linewidth is not modelled, rather than
picking a nonzero sigma that looks more physical and is calibrated to nothing.

---

## 8. Recorded for step 5, not built now: the polarizing beamsplitter

**A weak coherent pulse hitting a polarizing beamsplitter *splits*. It is not
routed.**

```
w_H, w_V = polarization_weights(signal.polarization)   # = (Tr(Pi_H rho), Tr(Pi_V rho))
mu_H = mu * w_H
mu_V = mu * w_V

P_H = 1 - exp(-eta_H * mu_H)      # independent
P_V = 1 - exp(-eta_V * mu_V)      # independent
```

Both detectors are evaluated, independently, every slot. **Double clicks therefore
occur at a real, calculable rate** — `P_H * P_V` for independent channels — and
that rate is a physical output of the model, not an artifact.

Worked example: `mu = 0.1`, `eta = 0.2`. An H-prepared pulse measured in the
diagonal basis gives `mu_H = mu_V = 0.05`, so `P_H = P_V = 1 - exp(-0.01) =
9.95e-3` and `P_both` is about `9.9e-5`. Measured in the matching basis,
`mu_V = 0` and `P_both = 0` exactly. The rate is basis-dependent, and it is one of
the things decoy BB84's security analysis needs to be right.

**The wrong model, stated here so step 5 does not rediscover it:** measuring a
polarization qstate, getting an outcome, and routing the whole pulse to the
corresponding detector. That is a single-photon model. It cannot produce a double
click at all, it makes the double-click rate identically zero at every `mu`, and
it silently assumes one photon per pulse — which is precisely the assumption decoy
BB84 exists to remove. Using it would make the simulator agree with the protocol's
own strawman.

Two further notes for that step:

- `ThresholdClickResolver(double_click_policy=...)` already exists and is where
  the *protocol's* response to a double click belongs (discard, random-assign, or
  count). The *rate* is physics and belongs in the PBS; the *response* is policy
  and belongs in the resolver. Do not conflate them.
- The PBS is a splitting component: **two** quantum egress ports carrying `mu_H`
  and `mu_V`. `EntangledPairSource` and the delay interferometer are both
  precedents; no new port machinery is needed.

---

## 9. What does not fit cleanly

Explained, not worked around.

**9.1 A Jones vector cannot represent depolarization.** `rotated_polarization` is
unitary, so channel polarization drift keeps the state pure and the model is
exact. Actual depolarization — the thing that raises QBER in a real polarization
link — needs a 2x2 density matrix, and the existing Kraus machinery in `qstate/`
is shaped `(2**arity, 2**arity)` for qubit *axes* and has no hook for a mode
descriptor. There is no clean home for it in this design. The honest V1 position:
model polarization drift, do not model depolarization, and say so in the run
report. `polarization_weights()` exists precisely so that the day a density matrix
arrives, one function changes and the PBS does not.

**9.2 `SinglePhotonDetector`'s dead-time model is coarse at GHz.**
`evaluate_window` blocks the *entire* arrival window if `time < dead_until` — a
documented stage-1 choice ("the whole arrival window is blocked. This keeps the
first implementation simple"). At a 1 ns slot period and a dead time of tens of
nanoseconds, a click blocks whole slots, which is roughly right; but the model
never recovers *mid-window*, so a detector that becomes live 10% into a window
stays dead for all of it. It biases detection rates low at high count rates. Not
worth fixing for DPS; worth knowing before anyone quotes a saturated-detector rate.

**9.3 The channel's phase noise is independent per pulse, and that is
pessimistic.** `alpha -> alpha * exp(i * delta_phi)` with an independent draw per
pulse gives the *differential* phase a variance of `2 * sigma_phi**2`. Real fiber
phase noise over a slot period of order a nanosecond is strongly correlated
between neighbours, so this over-estimates differential-phase error, and a
phase-encoded protocol's QBER reads pessimistic at any given `sigma_phi`. A Wiener
or Ornstein-Uhlenbeck drift is a different model with its own state. **Report the
discrepancy; never tune `sigma_phi` down to hide it.** Default
`phase_noise_stddev_rad = 0.0`, and any nonzero value carries this caveat into the
run report.

**9.4 The interferometer's N+1 slots have no clean protocol meaning.** The first
and last slots pair a real pulse with vacuum. Both detectors then fire with equal
probability, no bit is encoded, and Alice has no preparation to match. In a real
DPS system the train is effectively continuous and these are negligible; here they
are two of N. The agent must drop them explicitly with a counter. There is no way
to make them disappear that is also honest.

**9.5 `Signal.id` uniqueness across a re-emitting component.** The interferometer
creates new signals rather than transforming one, so it must namespace ids.
Nothing in the repo enforces `id` uniqueness — it is convention only — so a
collision would surface as a confusing agent-side mismatch rather than an error.
Worth one assertion in the interferometer's tests.

**9.6 `complex` in log metadata degrades to `repr()`.** `_to_jsonable`
(`tracing/sinks.py:197`) has no `complex` case, so `alpha` in a JSONL trace becomes
`"(0.447+0j)"`. Not a crash; not parseable either. The discipline is to log `mu`
and `phase_rad` as floats. A `complex` case in `_to_jsonable` would be a two-line
change, but it is `tracing/` code changed for one caller's convenience, and
`CLAUDE.md` invariant 7 says logging is observational — better to give it floats
than to teach it a new type.

**9.7 `gamma < 1` is reachable in V1 only through source timing jitter.** The
channel has a fixed delay and rejects jitter on the amplitude path (S5), and one
source means both arms share a `sigma`. So the only mechanism producing partial
overlap in a full run is the source's `timing_profile` shifting pulse *n* by
`delta_n`, giving `Delta_t = delta_n - delta_{n-1}` at the second beamsplitter.
Partial overlap is genuinely in scope and genuinely testable. **Two consequences,
recorded so neither is rediscovered:**

- **The `abs(gamma) = 0.6` to 20:80 result belongs in a `coherent_optics.py` unit test.**
  With equal amplitudes and zero differential phase, `interfere` returns
  `mu_0 : mu_1 = (1 - 0.6) : (1 + 0.6) = 20 : 80`. That is exact, needs no
  timeline, no source, and no seed, and it runs in microseconds. Asserting it from
  a full DPS run instead would make it a statistical claim requiring a 20-seed
  sweep to say anything — for a number that is not statistical at all.
- **A full run cannot reach `gamma < 1` by any other route.** If an end-to-end
  trial shows reduced visibility and the source is using `DeltaTiming`, the cause
  is *not* partial overlap — look at carrier phase (7.2) or channel phase noise
  (9.3) instead. Configuring a stochastic `timing_profile` is the only way to make
  `gamma < 1` happen in a run, and it does so through
  `Delta_t = delta_n - delta_{n-1}`, not through anything the channel does.

**9.8 The step-0 commit cannot be validated by the thing it exists for.** S1
through S11 add no behaviour that any existing test exercises; the 1460 stay green
because every change is defaulted-off. The only real guard is the
`_with_metadata` completeness test (S10). That is by design, and it is why that
one test is non-negotiable.

---

## 10. Where this design is more complex than the physics requires

Said directly, so it can be cut now rather than carried.

**10.1 `polarization` on `Signal` is dead weight until step 7.** DPS needs none.
COW needs none. The field costs a slot, a validation branch, and a
`_with_metadata` line that nothing reads.

Kept anyway, and here is the trade: settling `Signal`'s shape in one commit is the
entire point of this session. Adding the field at step 7 means a second shared
commit and a second round of "what else does this touch". The field is about 8
bytes per signal and one line; the *branch* — the channel's polarization noise
path, maybe 60 lines of arithmetic and a new RNG stream — is where the real cost
is, and **that is deferred to step 7** (S5). Deciding the shape and building the
branch are separable, and only the first has to happen now.

If you want to cut further: drop the field too, and accept one more shared commit
at step 7. That is a defensible call, and it is yours.

**10.2 Two carrier-phase policies where the physics needs one.**
`PerPulseRandomCarrierPhase` is not used by DPS. It exists to make the
DPS-destroying case testable, and because decoy BB84 will require it. Twelve
lines. If it feels like scaffolding: it is — but it is the scaffolding that turns
"`Theta` matters" from an assertion in this document into a failing test.

**10.3 The four-policy split may be one policy too many for V1.** With
`FixedIntensity` and `FixedCarrierPhase` as the only V1 implementations of two of
the four, those two are constants wearing a strategy interface. The alternative —
plain `mu: float` and `carrier_phase_rad: float` constructor fields now, promoted
to policies at step 7 — is genuinely simpler *today*. It is rejected because 7.2
shows the three protocols disagree about `Theta` specifically, so that seam has to
exist; and once one is a policy, making its three siblings match costs less than
explaining why they differ. But `FixedIntensity` alone would be a fair cut.

**10.4 `interfere` returning amplitudes rather than mean photon numbers is a
compromise.** The exact, defensible outputs are `mu_0` and `mu_1`. The output
*fields* at `abs(gamma) < 1` are truncations — a real port field is a superposition
of two non-identical envelopes, and the returned phase and width are the
interfering component's and the short arm's respectively. Returning amplitudes is
chosen only so the interferometer's outputs are ordinary `Signal`s that the
detector reads the same way it reads everything else. The cost is a footgun:
chaining a second interferometer would silently use a truncated field. It is
documented in the docstring and listed as a gap, which is the best available
answer short of a mode-decomposition model that this simulator has no business
carrying.

**10.5 What is *not* over-built, deliberately.** No separate modulator component
(section 5). No generic `PreparationSampler` (5.2). No `DetectorArray` amplitude
mode (step 5). No addition to `primitives/validation.py` (S2). No Poisson sampling
anywhere.

An earlier version of this list also claimed "no `CoherentState` class". That is
reversed after reading `scratch/optical.py` — see 11.1. It is the one verdict in
section 11 that runs the opposite way from the rest: something I had argued against
on principle turned out to be carrying weight I could not see without the code.

---

## 11. The reference implementation: what is carried over, and what is not

Read against the actual files this time, not against `CAPABILITY_MAP.md`'s account
of them: `scratch/optical.py` (575 lines), `weak_coherent_pulse_source.py` (475),
`phase_modulator.py` (598), `delay_interferometer.py` (1023), and
`dps-qkd-design-notes.md` (831). Four of the six steps were completed there, so
this is finished code, not a sketch.

The physics is excellent and most of it is adopted verbatim. The architecture is
rejected on four counts, all of them the ones named in the brief. Where a verdict
below reverses something earlier in this document, that is said outright.

### 11.1 Adopted, with the reasoning

**`CoherentState` as a value object — adopted, reversing section 4.** Section 4
originally specified a bare `complex` on `Signal` plus module functions, on the
grounds that a wrapper class "would add a construction site, an equality rule, and
a serialization question to buy nothing". Having read `scratch/optical.py:79-324`,
that was wrong. The class buys three things a bare `complex` does not:

- validation happens **once**, at construction (`bool` rejected, non-finite real or
  imaginary part rejected), instead of at the head of every function that takes an
  amplitude;
- `mean_photon_number` and `phase_rad` are `@property`, so call sites read
  `state.mean_photon_number` rather than `mean_photon_number(sig.amplitude)`;
- `interfere` can return a state whose μ is *not* `|α|²` — which, as 11.2 explains,
  it must.

The "only stored field" invariant is identical either way; the class does not
weaken it. `mean_photon_number` is computed as `re*re + im*im` rather than
`abs(alpha)**2`, avoiding the square-root round trip. Keep that.

Consequently `Signal` carries `coherent_state: CoherentState | None`, not
`amplitude: complex | None`. Section 2 is updated.

**The orthogonal-residual formulation of `interfere` — adopted, and it is better
than the equation section 4 specified.** Section 4 gave the μ equations and said
"returns the two output amplitudes, whose squared moduli are exactly those mu
values". `scratch/optical.py:544-567` instead splits the long arm into a component
along the short arm's mode (weight γ) and an orthogonal remainder:

```
mixed       = gamma * long.alpha
amp_0       = (short.alpha - mixed) / sqrt(2)
amp_1       = (short.alpha + mixed) / sqrt(2)
residual    = max(0.0, 1 - abs(gamma)**2) * long.mean_photon_number / 2
mu_k        = |amp_k|**2 + residual
```

I checked this reduces to the stated equation: expanding `|amp_0|²` gives
`½(|α_s|² + |γ|²|α_ℓ|² − 2Re(α_s*γα_ℓ))`, and adding
`½(1−|γ|²)|α_ℓ|²` yields `½(|α_s|² + |α_ℓ|²) − Re(α_s*α_ℓγ)` exactly. Same
physics, but the non-interfering part of the light is *visible in the code* rather
than folded into an algebraic identity. Adopt it.

It also forces a correction to section 4. Because the residual is added after the
amplitude is formed, the returned state cannot simply carry `amp_k` — its modulus
is wrong. The reference reconstructs with
`CoherentState.from_mean_photon_number(mu_k, phase_rad=cmath.phase(amp_k))`, which
round-trips μ through `sqrt`. **So μ does not come back bit-exact**: 0.2 returns as
0.19999999999999998, and an analytically dark port lands near 1e-16 rather than 0.
Section 4's "squared moduli are exactly those mu values" was wrong and is fixed.

**`vacuum_like(other)` — adopted.** `delay_interferometer.py:247-273` returns a
vacuum contribution that borrows the present arm's BS2 tick, σ, and wavelength, and
carries `pulse_index=None`. Because the interference term is proportional to *both*
amplitudes, a vacuum partner kills it at any γ, so the first pulse, the last pulse,
and the flush all go through the identical `interfere()` call with no branch. It
also collapses the output-σ rule to one line. This is the single most elegant thing
in the reference and section 6's step 4 now specifies it.

**Combine-on-arrival, and the causality deferral.** `_handle_arrival`
(`delay_interferometer.py:628-682`) takes the holder before doing anything
(`held, self._held = self._held, None`), combines or defers, then refills it with
this pulse's own long arm. Two findings I would not have reached myself:

- *Why not schedule a BS2 event per arm.* At the design point τ = T, pulse *k*'s
  short arm and pulse *k−1*'s long arm land on **the same tick**. Any scheme that
  processes them as two events makes correctness depend on same-tick ordering.
  Combine-on-arrival makes it one event and the contest cannot arise.
- *The causality fix.* Combine-on-arrival is acausal when the new pulse arrives
  **before** the held arm has reached BS2 — reachable whenever the source uses a
  stochastic timing profile, which shortens spacing below τ. The guard is
  `if partner.bs2_tick <= arrival_tick: resolve else: defer`, deferring to
  `ACTION_RESOLVE_BS2` at the long arm's BS2 tick and releasing the holder in the
  same step. Rejecting early arrivals was considered and rejected there, correctly:
  it turns a legitimate timing profile into a seed-dependent mid-run abort.

**Determinism finding: adding an RNG stream cannot perturb existing ones.**
`engine/rng_manager.py:338-341` builds each stream as
`SeedSequence(entropy=root.entropy, spawn_key=_path_to_spawn_key(path))` — derived
from the stream *path*, never from creation order or stream count. I verified this
in the source. It is what makes S5's two new channel streams safe, and it deserves
to be cited rather than re-derived every time someone adds a stream.

**`attenuated` rather than `with_attenuation`.** The naming asymmetry with
`with_phase_shift` is deliberate: `with_attenuation(0.1)` does not say whether 0.1
is the loss or what survives it. Keep the asymmetry and the reasoning.

**Numerical honesty, three findings, all adopted into section 6's test list:**

- `exp(1j*pi)` is `(-1+1.2246e-16j)`, so `theta = pi` does not give exactly `-alpha`.
  Deliberately not special-cased, because a hidden branch for π would make
  exactness look guaranteed and the dark port would rest on an implementation
  accident. A perfect dark port therefore delivers μ ≈ 1e-33, **not 0** — step 5's
  detector must produce "no click" from its own statistics on a tiny-but-nonzero μ.
- Attenuation and phase shift commute analytically but **not bit-exactly**;
  complex multiplication is not associative in floating point. An earlier draft
  there claimed exact commutation and a test caught it.
- `min(int(rng.random() * n), n - 1)` guards the `random() == 1.0` boundary in
  `RandomPhaseChoice.select_phase`. Small, and exactly the kind of thing that is
  wrong in a reimplementation.

**σ is not converted with `seconds_to_ticks`.** That helper rounds to integer
picoseconds, which would quantize γ and make a partial-overlap test unwritable at
small Δt. `CLAUDE.md`'s conversion rule governs event *times*, which must be
integers; a continuous width belongs to `duration_s`'s family. Section 2 now says so.

**Append new `Signal` fields last, never grouped by meaning.** The reference
appended `coherent_state`, then `temporal_mode_sigma_s`, each at the end, and left
a comment saying not to tidy them into place. The reason is better than "avoid
breaking positional callers": appending makes the question *unanswerable* rather
than answered once, including for a call site added later. Section 3's S1 changed
from "insert after `wavelength_nm`" to "append last".

**The selector pattern, generalised to four policies.** `PhaseSelection(phase_rad,
index)` is a frozen record carrying both the value and its position in the
alphabet, and `select_phase(index, rng)` is **pure** — it receives the caller's
counter instead of holding a cursor, so one selector can drive several components
without shared hidden state. `PhaseSequence(repeat=False)` raises on exhaustion
rather than wrapping, because a pattern shorter than the run would otherwise
produce a plausible-looking but wrong key. The stated precedent is `GateModel` in
`detectors/primitives/gate.py` — a Protocol with a trivial implementation
(`AlwaysOpenGate`), a parametric one (`PeriodicGate`), and an explicit-sequence one
(`ScheduledGate`) — not `EmissionTimingProfile`. All of this transfers directly onto
section 5's four policies; section 5 is updated to return `(value, index)` records
rather than bare values.

**Verified repo findings worth keeping regardless of architecture.** Each cost real
effort and none is re-derivable cheaply:

- `_is_before_stop` is **dead code in all three source components** — defined,
  never called, because `schedule_next_emission_event` does the check internally.
- The stop-tick convention is **exclusive**, and the double check in
  `schedule_next_emission_event` drops **only the final slot's** pulse: since the
  scheduler enforces `delay < period`, `slot_k + delay >= stop` implies
  `slot_{k+1} >= stop` already. An earlier note in the project claimed otherwise.
- `SubsystemHandle.kind` accepts `"mode"` but nothing in `src/` constructs or reads
  it. Deliberately unused by a coherent pulse: a populated `state_targets` beside
  `state_ref=None` would mislead a reader into expecting a qstate.
- The standard debug counters all assume qstate-backed signals, so
  `timeline.qstate.size()` stays 0 on a coherent run. The source exposes its own
  `pulse_count`.
- `tests/signal/` exists but has **no `__init__.py`**, so it is not a package and
  `from ..support ...` fails from inside it. Confirmed on disk. Any new test folder
  that imports shared scaffolding needs the `__init__.py`.
- `docs/dev/count_code_lines.py` is the line counter, and it asserts
  `code + docstring + comment + blank == total`. The memo records that an earlier
  ad-hoc counter double-subtracted blank lines inside docstrings and understated
  every figure it produced, with nothing catching it because nothing checked the
  sum. Use the script; do not write another one inline.

### 11.2 Two corrections this pass forced on *this* document

**`_with_metadata` omission raises `AttributeError` — it is not silent.** Section
1.1 called it "silent data loss ... just `amplitude=None` on the far side". That is
wrong, and the memo is right: `Signal` is `slots=True`, so a field never
`object.__setattr__`-ed is *unset*, and the first read raises
`AttributeError: 'Signal' object has no attribute ...`. I confirmed this with a
minimal frozen-slots dataclass. The fix is still required — a crash on every
channel crossing is not an acceptable failure mode either — but the severity
framing was wrong and section 1.1 is corrected.

**`_derived` is a better fix than "remember to add the field to `_with_metadata`".**
The reference replaced the hand-copy with `Signal._derived(...)`, which iterates
`_SIGNAL_FIELD_NAMES` read from `dataclasses.fields(Signal)` and replaces only what
the caller names, keeping `_with_metadata` as a delegating wrapper. Three details
make it work, all adopted into S3:

- a `_KEEP` sentinel rather than `None`, because `None` is a legal value for
  `coherent_state` — clearing an amplitude and preserving one must stay
  distinguishable;
- **not** a fresh `Signal(...)` at the transform site: that fails *silently*,
  giving a newly added field its declared default at every construction site with
  nothing raising, which is strictly worse than the `AttributeError`;
- **not** a second `_with_coherent_state` sibling: the channel mutates amplitude
  *and* metadata, so two siblings means two hand-copies to keep in sync, of which
  the guard tests covered only one.

Measured cost, from the memo: the loop is 1.5× the hand-copy (882 ns → 1287 ns),
which is ~8% of one full `QuantumChannel._transmit_now` hop (5736 ns); the copy
itself is 24% of a hop. A precomputed `(get, set)` variant recovers most of it
(974 ns, 1.10×) while still deriving its field list from `fields(Signal)`. Not
taken, correctly — nothing profiles signal copying as a bottleneck.

This supersedes S3 as originally written. The completeness test in S10 stays: it
now asserts that `_SIGNAL_FIELD_NAMES` still equals `fields(Signal)`, which is the
property the whole scheme rests on.

### 11.3 Rejected — the four architectural points, plus two more

**1. The source stores one shared `CoherentState` and has no path to polarization.**
`weak_coherent_pulse_source.py:227-229` builds the state once in `__post_init__`
from a scalar `mean_photon_number` field, and `_emit_now` attaches *that exact
object* to every pulse — the tests assert identity with `is`. The memo calls this
"non-sampling is structural", and as a guarantee that the amplitude never drifts it
is genuinely elegant. But it makes the source a fixed-amplitude pulse-train
generator: there is no per-pulse μ, no per-pulse phase, no polarization, and
exactly one RNG stream (`"timing"`). Decoy states and BB84 polarization are not
extensions of this shape, they are rewrites of it. Rejected, and this is the single
biggest structural difference in the document.

What survives is the **entire emit skeleton**, which is excellent and adopted
essentially verbatim: reuse of `_common.py`'s chained scheduler, one `pulse_index`
counter with no `attempt_index`/`emission_index` split (every active slot emits),
`state_ref=None`, `validation_flag=False` on the trusted hot path, and the log /
report / transmit tail. The diff against the reference is small and local: replace
the instance attribute with four policy draws and build the state per pulse.

**2. The modulator is a separate component.** `phase_modulator.py` is a full
component — 598 lines, its own package, its own action, its own report, its own RNG
stream, its own port pair — whose entire physical content is
`state.with_phase_shift(theta)`. It exists because the source could not vary
anything per pulse (point 1), so the per-pulse choice had to happen *somewhere*.
Fix point 1 and the component's reason to exist disappears. Rejected; the phase
choice moves inside the source, where μ, Θ, and polarization already have to be
chosen anyway.

The memo's own text is the tell: the applied θ "is **not** recoverable from the
amplitude", so step 6 must read it from `PhaseModulationReport.phase_rad`. The
modulator's only durable output is a report describing a choice — and in this
design the source already emits a report describing its choices.

Two things from it are kept even though the component is not: the selector pattern
(11.1) and the observation that `store_source_report` stamps
`report_kind="source_preparation"` and would misdescribe any non-source, which is
why the modulator needed a private `_store_report`.

**3. One `optical.py` under `signal/`.** The reference put the type and all the
arithmetic together at `src/simyuj/signal/optical.py`. Rejected on both counts —
the single module and its location — and the replacement is a two-way split that
building commit 1 forced (section 4):

- the **type** goes to `primitives/coherent_state.py`, following
  `SubsystemHandle`, which is the same kind of value carried by the same envelope
  and already lives there;
- the **arithmetic** goes to `components/coherent_optics.py`, because optical
  operations are component-layer physics and `signal/`'s own module docstring
  disclaims owning math.

Keeping both under `signal/` would make `Signal` the home of the transport
definition *and* the physics; keeping both under `components/` is a circular
import. The module's *contents* still move essentially unchanged — this is a
relocation and a split, not a rewrite.

**4. The channel treats the two payloads as mutually exclusive.** The reference's
branch (memo, step 3) is a single early-return inserted before the qstate path:

```python
coherent_state = signal.coherent_state
if coherent_state is not None:
    self._transmit_coherent_now(...)
    return
```

This has a real virtue I want to record because rejecting it has a cost: the entire
qstate path stayed **byte-identical**, verified by diffing the old and new line
ranges rather than by reading hunk headers. That is a strong review property and my
restructuring into three independent branches gives it up.

It is still rejected. `if coherent: ...; return` cannot express a pulse that carries
both an amplitude and a polarization descriptor — the polarization branch would sit
downstream of a `return`. The reference had no polarization, so the early return
cost it nothing; here it forecloses decoy BB84. S5's three-branch form is kept, and
the cost is stated: the qstate path is rewritten rather than preserved verbatim, so
its regression tests carry more weight than they did.

The three rejections inside that branch are adopted wholesale — jitter, `noise_models`,
and `state_ref`-with-amplitude, all as event-time `ValueError`s because the channel
cannot know at construction which payload it will carry. Two details worth keeping:
the jitter rejection is backed by an **observed** reordering (at seed 11, signal
`s1` is delivered at tick 12 and `s0` at tick 14), not just an argument; and the
three live in one `_require_coherent_transport_supported` method so they read as a
rule rather than as accumulated caution.

**5. `CoherentPulsePreparationReport` — rejected as built, adopted as a shape, and
it replaces S9.** The reference added a sibling report class carrying the shared
`CoherentState`, leaving `SourcePreparationReport` completely untouched and only
widening `store_source_report`'s annotations. That is *less* shared-code change
than S9's "make `SourcePreparationReport.state_ref` optional", and it is more
honest — a report with `sampler_index`/`sampler_label` for a source that has no
sampler is a lie. Adopted.

What is rejected is its content: it records only the one shared state, because
there was nothing per-pulse to record. Here it must carry the four choices as typed
fields (μ, Θ, φ_enc, polarization, each with its alphabet index). **S9 is therefore
deleted from the shared-code list** — `sources/reports.py` gains a class and one
widened annotation, and no existing type changes at all.

**6. "Two quantum egress ports had no precedent" — false, and it was asserted as
AST-verified.** The memo (step 4, "Corrections to earlier notes") states this was
"verified by an AST scan of every `Port(...)` construction in `src/`", and concludes
`QuantumMemory` is the only component with two egress ports at all. It is wrong:
`EntangledPairSource` constructs two ports through `quantum_output_port`
(`entangled_pair_source.py:207` and `:212`), and that helper hard-codes
`PortKind.QUANTUM` with `PortDirection.EGRESS` (`sources/_common.py:570`). Two
quantum egress ports on one component, in `src/`, predating the whole effort.

`CAPABILITY_MAP.md` inherited the same claim and has been corrected. I flag it not
because it changes any decision — the conclusion drawn from it, that the port layer
needs no change, is correct anyway — but because it is the one falsifiable claim in
the memo that carries an explicit verification method and fails it. Everything else
in the memo that I could check independently held up.

### 11.4 Not evaluated

`delay_interferometer.py`'s report/logging blocks, `_resolve`'s 101 lines, and the
`InterferenceReport` field set were skimmed, not audited. The memo's own guidance —
"do not inherit the per-method log block", justified because the interferometer
deliberately does not validate τ against the pulse period and so a mismatch is
visible only in the run record — is adopted as written, and section 6's step 4
repeats it. The 544-line breakdown in the memo is a useful size expectation for
whoever builds it.
