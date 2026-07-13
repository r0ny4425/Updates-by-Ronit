# Overview

This example isolates the classical BB84 post-processing stage.

Alice and Bob start with arrays:

```text
Alice bases, Alice measured bits
Bob bases, Bob measured bits or missed detections
```

The agents then talk over two public classical channels. Every stage is driven
by messages or local scheduled events.

## What the example does

1. Alice sends her basis list.
2. Bob keeps positions where the bases match and he has a measured bit.
3. Bob sends the sifted indices back to Alice.
4. Alice chooses a public sample for QBER estimation.
5. Bob computes the sample error rate.
6. Alice and Bob remove the sample bits from the remaining key material.
7. Alice starts Cascade with a block size chosen from the estimated QBER.
8. Bob asks Alice parity questions until Cascade finishes.
9. Bob sends a Toeplitz verification tag.
10. Alice checks the tag.
11. Bob sends a Toeplitz privacy-amplification seed.
12. Both sides hash down to the final key.

The message names are shown in
[the message-flow diagram](assets/postprocessing_message_flow.svg).

For the later stages, see:

- [Cascade reconciliation](02_cascade.md)
- [Verification and privacy amplification](03_verification_and_privacy.md)

## What the example does not do

This folder does not simulate photons, channels, detector timing, dark counts,
or slot assignment. That is handled by the full single-photon BB84 example in
`examples/bb84`.

This example also does not claim a full finite-key security proof. The final
key length uses a teaching budget:

```text
estimated QBER
statistical margin
Cascade leakage
verification leakage
security margin
```

That is useful for learning how the pieces connect, but it should not be
presented as a production QKD security analysis.

## How to read the code

Read `agents.py` by following message types.

| Message | Direction | Purpose |
| --- | --- | --- |
| `sift.bases` | Alice to Bob | Alice reveals bases. |
| `sift.indices` | Bob to Alice | Bob returns positions that survive sifting. |
| `estimate.sample` | Alice to Bob | Alice reveals sample positions and bits. |
| `estimate.result` | Bob to Alice | Bob returns QBER and accept/reject result. |
| `cascade.start` | Alice to Bob | Alice starts reconciliation parameters. |
| `cascade.parity_request` | Bob to Alice | Bob asks for a parity check. |
| `cascade.parity_response` | Alice to Bob | Alice answers the parity check. |
| `verify.tag` | Bob to Alice | Bob sends a hash tag for verification. |
| `verify.result` | Alice to Bob | Alice says whether verification passed. |
| `privacy.seed` | Bob to Alice | Bob sends the privacy-amplification seed. |
| `finished` | Alice to Bob | Alice confirms the final key length. |
| `abort` | either direction | One side stops the run and gives a reason. |

Local events are used when one side needs to keep progressing without a new
external message. For example, Alice schedules `estimate.choose_sample`, and
Bob schedules `cascade.next_request`, `cascade.completed`, and
`privacy.prepare`.

## What to check

Useful summary fields:

```text
sifted bits
matched detections rejected
estimated qber
cascade parity requests
cascade corrections
cascade leaked bits
verification tag bits
demo entropy budget bits
final key bits
alice final key == bob final key
```

An abort is not always a bug. It may mean there were too few sifted bits, the
sample QBER was too high, verification failed, or the final key budget was too
small.
