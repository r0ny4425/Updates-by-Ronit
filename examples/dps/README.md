# DPS-QKD — Alice's transmitter

Differential-phase-shift QKD, built one stage at a time. **This package is
currently stage 1 only: the transmitter.** There is no receiver and no protocol
agent yet.

```bash
uv run python -m examples.dps.demo --num-slots 12 --show-pulses 6
```

## What runs

```
WeakCoherentPulseSource  --out-->  PulseTap
        |
     report port  (unwired in this stage; reports are read from source.reports)
```

Once per slot the source chooses a mean photon number, a carrier phase, and an
encoding phase, builds

```
alpha = sqrt(mu) * exp(i * (Theta + phi_enc))
```

and emits a `SignalKind.PULSE` signal carrying that `CoherentState`. Every
choice is recorded in a `CoherentPulsePreparationReport`.

`PulseTap` is a stand-in receiver. The delay interferometer replaces it at step
4 of `docs/dev/dps-design.md`; until then a coherent pulse has nowhere else to
go, because `QuantumChannel` resolves qstate targets unconditionally and rejects
a signal that carries none.

## Counters worth reading

| Counter | Should be |
|---|---|
| `pulses_emitted` | exactly `configured_slots` — there is no emission Bernoulli |
| `pulses_delivered` | equal to `pulses_emitted` — nothing lossy is wired in yet |
| `qstate_records` | **0** — a coherent pulse creates no quantum state |
| `differential_bits` | `pulses_emitted - 1` — the first pulse has no predecessor |
| `encoding_phase_histogram` | roughly balanced over `(0, pi)` |
| `carrier_phase_distinct_values` | `1` by default; `pulses_emitted` with `--randomize-carrier-phase` |

`qstate_records: 0` is the one to look at first. Nothing anywhere samples
`n ~ Poisson(mu)`; photon statistics are integrated in closed form at detection,
which does not exist yet.

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

The demo prints this list on every run. Finite laser linewidth (`FixedCarrierPhase`
means infinite coherence length), modulator insertion loss and extinction ratio,
chirp and pulse broadening, and the entire receive path.

## Files

| File | Holds |
|---|---|
| `configs.py` | frozen dataclasses, physical units, and the encoding alphabet |
| `helpers.py` | pure functions: slot arithmetic, differential-bit decoding |
| `trial.py` | explicit wiring, `PulseTap`, the run, the counters |
| `reporting.py` | summary text and the JSON report |
| `demo.py` | CLI entry point |

`agents.py` arrives at step 6, once there is something to receive.
