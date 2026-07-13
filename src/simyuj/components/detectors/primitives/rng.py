"""RNG stream bundle records for detector-channel sampling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetectorRNGStreams:
    """
    Deterministic RNG stream bundle for one detector channel.

    Parameters
    ----------
    efficiency : object
        RNG used for Bernoulli signal-detection decisions when detector
        efficiency is strictly between zero and one.
    dark : object
        RNG used for Poisson dark-count sampling and optional time-resolved
        dark-count offsets.
    jitter : object
        RNG used for detector response jitter when jitter standard deviation is
        non-zero.
    afterpulse : object or None, default=None
        RNG used for afterpulse occurrence and timing when afterpulse
        probability is non-zero.

    Notes
    -----
    Component ``bind`` methods create these streams from ``Timeline.rng(...)``
    before event execution. Fixed-probability edge cases such as efficiency
    ``0.0`` or ``1.0`` may avoid consuming the corresponding stream. The fields
    stay typed as ``object`` because each consuming model requires different
    RNG methods, such as ``random()``, ``poisson()``, or ``normal()``.
    """

    efficiency: object
    dark: object
    jitter: object
    afterpulse: object | None = None


__all__ = [
    "DetectorRNGStreams",
]
