# DPS-QKD — transmitter, fibre, and detecting receiver

Differential-phase-shift QKD, built one stage at a time. **Alice sends, Bob
detects, and the run reports a QBER.** What is still missing is the *protocol*:
there is no classical channel and no agents, so Alice's bits are read from her
own reports rather than learned from a message.

```bash
uv run python -m examples.dps.demo --num-slots 12 --show-pulses 6
```

## What runs

```
WeakCoherentPulseSource --> QuantumChannel --> DelayInterferometer --> DetectionCollector
                                                  |  d0 -> out_0        (detection port)
                                                  |  d1 -> out_1
                                                  +--> out_0, out_1 --> PulseTap x2
```

Bob's two detectors live *inside* the interferometer: a DPS receiver is one
physical unit, and BS2 produces both output amplitudes in one call, so one slot
decision follows with no arrival buffering. The optical ports stay wired to taps
because the device always puts light on both — the destructive one carrying
nearly nothing is a result, not an absence — and because keeping the exact
intensities reachable is what lets the detector be checked against the ideal
readout.

Once per slot the source chooses a mean photon number, a carrier phase, and an
encoding phase, builds

```
alpha = sqrt(mu) * exp(i * (Theta + phi_enc))
```

and emits a `SignalKind.PULSE` signal carrying that `CoherentState`. Every
choice is recorded in a `CoherentPulsePreparationReport`.

Each interference slot becomes one `DetectionReport`. Bob's bit is the port
that fired: `out_0` is bit `1`, `out_1` is bit `0`. A slot can also produce no
click at all, or a double click, and the summary counts all three.

**Detector efficiency is not the probability of a click.** For a coherent pulse
that is `1 - exp(-eta * mu)`, so at the default `eta = 0.6`, `mu = 0.2` a bright
port fires on about 11% of slots — and a *perfect* detector would still only
reach 18%, because most pulses contain no photon at all. Dead time then costs
about a third of what is left.

## Counters worth reading

| Counter | Should be |
|---|---|
| `pulses_emitted` | exactly `configured_slots` — there is no emission Bernoulli |
| `pulses_delivered` | equal to `pulses_emitted` — nothing lossy is wired in yet |
| `qstate_records` | **0** — this trial configures no polarization, so nothing reaches qstate |
| `differential_bits` | `pulses_emitted - 1` — the first pulse has no predecessor |
| `encoding_phase_histogram` | roughly balanced over `(0, pi)` |
| `carrier_phase_distinct_values` | `1` by default; `pulses_emitted` with `--randomize-carrier-phase` |

`qstate_records: 0` is the one to look at first, though read it for what it is:
a statement about *this configuration*, not about coherent pulses in general. A
source given a polarization selector prepares one record per pulse for the mode
its amplitude occupies, and this counter would then track the pulse count. The
claim that holds either way is the one underneath: nothing anywhere samples
`n ~ Poisson(mu)`; photon statistics are integrated in closed form at detection,
as `1 - exp(-eta*mu)` — one uniform draw against a probability, never a count.

### And the receiver's

| Counter | Default run |
|---|---|
| `edge_slots_dropped` | exactly `2` — the first pulse's short arm and the flush each meet vacuum |
| `slots_with_click` + `slots_no_click` + `slots_double_click` + `edge_slots_dropped` | `interference_slots` |
| `qber` | **`0.0`** — lossless, noiseless, and dark counts are 5e-8 per slot |
| `clicks_per_pulse` | about `0.072` |

The QBER is exactly zero rather than merely small, and that is a statement about
the wiring rather than a claim about physics: what limits it is dark counts, and
at 100 Hz against a 500 ps window there are effectively none. Turn them up and
it moves.

## Each imperfection breaks something different

| Flag | Clicks | QBER |
|---|---|---|
| `--channel-attenuation-db-per-km` | **down** | unchanged |
| `--channel-phase-noise-rad` | unchanged | **up** |
| `--dark-count-rate-hz` | **up** | **up** |
| `--detector-efficiency` (lower) | **down** | unchanged |

Loss scales both interferometer arms together, so it never moves light to the
wrong port — it costs signal, not key. Phase noise redistributes light *between*
the ports without destroying any, so the same slots click and more of them click
on the wrong side. A dark count fires a port the light did not, so it adds a
click and gets it wrong half the time.

## `--randomize-carrier-phase`

Draws an independent carrier phase per pulse. This is what decoy-state BB84
needs and what DPS cannot survive: the differential phase picks up
`Theta_n - Theta_{n-1}`, itself uniform, so once the receiver exists the
visibility goes to zero. The flag is here so that failure is reproducible rather
than merely described.

## The alphabet trap

`DPS_ENCODING_PHASES` in `configs.py` is defined once, and both the source
configuration and `helpers.dps_differential_bit` are written against that one
constant. Index `0` is phase `0`; index `1` is phase `pi`.

Reorder it and **every differential bit inverts silently** — no exception, and
the run still produces a plausible key that is wrong. The report records
`encoding_phase_rad` beside `encoding_phase_index` precisely so a consumer can
check, but nothing checks automatically. Do not inline the alphabet elsewhere.

## What is not modelled

The demo prints this list on every run. Finite laser linewidth
(`FixedCarrierPhase` means infinite coherence length), modulator insertion loss
and extinction ratio, chirp and pulse broadening, channel realism beyond a delay
plus uniform attenuation plus per-pulse phase noise, photon arrival time *within*
the pulse envelope, polarization-resolved detection, photon-number resolution,
interferometer non-idealities, and the whole protocol layer.

## Files

| File | Holds |
|---|---|
| `configs.py` | frozen dataclasses, physical units, and the encoding alphabet |
| `helpers.py` | pure functions: slot arithmetic, differential-bit decoding |
| `trial.py` | explicit wiring, `DetectionCollector`, the slot reader, the counters |
| `reporting.py` | summary text and the JSON report |
| `demo.py` | CLI entry point |

`agents.py` arrives at step 6: Bob announcing detection *times* on a classical
channel, Alice mapping each announced slot to `phi_n - phi_{n-1}`, and the error
correction and privacy amplification that turn a sifted key into a secret one.
