# Single-photon BB84 example

This example runs a two-node BB84 protocol in the SimYuj event engine. It is
meant to be read as code, not only executed as a script.

Alice owns a single-photon source. Bob owns a detector array. A lossy quantum
fiber carries the optical signals from Alice to Bob. Two public classical
channels carry the post-processing messages. The run continues through
sifting, QBER estimation, Cascade reconciliation, verification, and privacy
amplification.

![BB84 network](docs/assets/bb84_network.svg)

## Run it

From the repository root:

```bash
python examples/bb84/demo.py
```

With a shorter run:

```bash
python examples/bb84/demo.py --distance-km 10 --num-slots 5000 --seed 2027
```

Save the event log and final JSON report:

```bash
python examples/bb84/demo.py \
  --distance-km 25 \
  --num-slots 30000 \
  --log-file /tmp/bb84_events.jsonl \
  --report-file /tmp/bb84_report.json
```

The command-line interface exposes the common knobs. For detailed device
settings, call `run_bb84_trial` from Python and pass override dictionaries:

```python
from examples.bb84 import run_bb84_trial

trial = run_bb84_trial(
    distance_km=25,
    master_seed=2027,
    source_overrides={
        "emission_probability": 0.65,
        "timing_jitter_stddev_s": 30e-12,
    },
    quantum_channel_overrides={
        "attenuation_db_per_km": 0.22,
        "fixed_insertion_loss_db": 3.0,
        "depolarizing_probability": 0.01,
    },
    detector_overrides={
        "efficiency": 0.75,
        "dark_count_rate_hz": 500.0,
    },
)
```

The override keys are the field names in `configs.py`.

## What this example does

This is an event-based simulation. The source, quantum channel, detector,
agents, and classical messages are all wired into a `SessionRuntime`.

The main steps are:

1. Alice's source schedules emission slots.
2. Prepared photons travel through a lossy quantum channel.
3. Bob's detector measures incoming signals in a random BB84 basis.
4. Alice records source reports by slot.
5. Bob records detector reports by arrival time and assigns them to slots.
6. Alice announces that the quantum transmission frame is finished.
7. Bob starts sifting after a detector guard time.
8. Alice and Bob estimate QBER from a public sample.
9. Bob drives Cascade reconciliation by asking Alice parity questions.
10. Bob sends a verification hash.
11. Bob sends a privacy-amplification seed.
12. Both sides derive the final key.

The event flow is shown in more detail in
[docs/02_event_flow.md](docs/02_event_flow.md).

## What this example does not claim

This is not a production QKD security proof.

The simulation includes realistic engineering pieces such as fiber loss,
source inefficiency, detector efficiency, detector failures, timing jitter,
dead time, afterpulsing parameters, and a depolarizing channel model. It also
includes event-based public-channel post-processing.

The public classical channel is treated as reliable and authenticated. The
privacy-amplification length is a teaching-level budget based on the estimated
QBER, revealed bits, and a safety margin. It is useful for learning and for
testing simulator structure, but it should not be presented as a full
finite-key security analysis.

## File map

```text
examples/bb84/
    configs.py      device and post-processing settings
    helpers.py      small BB84, timing, and message helpers
    agents.py       Alice and Bob protocol agents
    trial.py        builds the network and runs one trial
    reporting.py    report writing and short summaries
    demo.py         command-line entry point
    docs/           explanation and diagrams
```

The notebook in `tutorials/protocols/bb84.ipynb` imports this package. The
notebook is for exploration; this folder is the runnable example code.

## Where to start reading

If you are new to the simulator, read in this order:

1. [Overview](docs/01_overview.md)
2. [Event flow](docs/02_event_flow.md)
3. [Writing your own protocol](docs/03_write_your_own_protocol.md)

Then open `trial.py`. It shows how devices, agents, links, and ports are wired
together. After that, read `agents.py`, where the protocol state machine lives.
