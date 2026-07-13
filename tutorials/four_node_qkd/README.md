# Concurrent four-node BB84 and E91 tutorial

This tutorial demonstrates two complete QKD workflows sharing one simulated
network and one event timeline:

```text
A -- single photons / BB84 --> B

                 +----------> B
C -- psi- pairs -|
                 +----------> D

             E91 key: B <--> D
```

Node B owns separate BB84 and E91 detectors and protocol agents. Both sources
start at tick zero, so BB84 transmission and post-processing overlap the E91
quantum frame. Every link is a real `QuantumChannel` or `ClassicalChannel`;
the tutorial does not use channel, source, detector, or agent stubs.

## Run it

From the repository root, using the project virtual environment:

```bash
.venv/bin/python - <<'PY'
from tutorials.four_node_qkd import (
    run_four_node_qkd_trial,
    summarize_four_node_trial,
)

result = run_four_node_qkd_trial()
print(summarize_four_node_trial(result))
PY
```

For the guided explanation, run `four_node_bb84_e91.ipynb` from top to bottom.
The default physical scenario is deliberately large enough to produce
non-empty final keys under the fixed seed.

## What is simulated

The A-B BB84 session reuses the complete reference implementation in
`examples/bb84`: physical preparation and detection, timing-based slot
assignment, sifting, QBER sampling, Cascade reconciliation, verification, and
Toeplitz privacy amplification.

For E91, C emits `psi-` pairs through independent 10 km fibers to B and D. The
receivers choose among three random analyzer settings. Their agents then use
physical public fibers for:

1. coincidence and basis sifting;
2. CHSH sample disclosure and evaluation;
3. public QBER sampling;
4. interactive Cascade parity queries;
5. Toeplitz verification;
6. seeded Toeplitz privacy amplification;
7. final-key length acknowledgement.

The source at C only announces that its quantum frame has ended. It never
receives the raw, reconciled, or final B-D key.

## Customize it

All physical and E91 post-processing settings are immutable dataclasses. Use
`dataclasses.replace` to change only the required values:

```python
from dataclasses import replace

from tutorials.four_node_qkd import FourNodeQKDConfig, run_four_node_qkd_trial

base = FourNodeQKDConfig(master_seed=2030)
custom = replace(
    base,
    e91_c_to_b=replace(
        base.e91_c_to_b,
        length_m=20_000,
        depolarizing_probability=0.05,
    ),
)
result = run_four_node_qkd_trial(custom)
```

Public classical links intentionally require zero loss. They still model
distance-dependent propagation delay, but retransmission and authentication
key consumption are outside this tutorial.

## Security boundary

The E91 extraction length is an auditable teaching budget using observed CHSH,
QBER, actual Cascade disclosures, a verification tag, and a security margin.
The asymptotic CHSH term follows Pironio et al., *Device-independent quantum key
distribution secure against collective attacks*:

<https://arxiv.org/abs/0903.4460>

This tutorial postselects coincident detections and uses fixed safety margins.
It therefore does **not** close the detection loophole and does not claim a
composable finite-key or production device-independent security proof.

## File map

```text
configs.py      physical and protocol settings
helpers.py      basis, CHSH, timing, and privacy-budget helpers
e91_agents.py   event-driven C, B, and D E91 agents
trial.py        physical network construction and execution
reporting.py    compact console and JSON summaries
```
