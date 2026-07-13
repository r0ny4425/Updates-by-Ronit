"""
Unit aliases and time conversion helpers.

Simulation time in the engine is represented as integer ticks.
The default repository-wide physical mapping is:

- 1 tick == 1 picosecond

Timeline and Event remain generic integer-tick engine primitives; this module
defines how component models translate those ticks to physical seconds.
All seconds-to-unit conversions use Python's ``round``; fractional half values
therefore follow Python's nearest-even rounding behavior.
"""

from __future__ import annotations

from typing import NewType, TypeAlias

SimTick = NewType("SimTick", int)
Hertz: TypeAlias = float

DEFAULT_TICK_PS = 1
DEFAULT_TICK_SECONDS = 1e-12


def seconds_to_ps(seconds: float) -> int:
    """Convert seconds to picoseconds.

    Parameters
    ----------
    seconds : float
        Non-negative duration in seconds.

    Returns
    -------
    int
        Duration rounded to the nearest picosecond.

    Raises
    ------
    ValueError
        If `seconds` is negative.
    """
    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    return round(seconds * 1_000_000_000_000)


def seconds_to_ns(seconds: float) -> int:
    """Convert seconds to nanoseconds.

    Parameters
    ----------
    seconds : float
        Non-negative duration in seconds.

    Returns
    -------
    int
        Duration rounded to the nearest nanosecond.

    Raises
    ------
    ValueError
        If `seconds` is negative.
    """
    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    return round(seconds * 1_000_000_000)


def seconds_to_us(seconds: float) -> int:
    """Convert seconds to microseconds.

    Parameters
    ----------
    seconds : float
        Non-negative duration in seconds.

    Returns
    -------
    int
        Duration rounded to the nearest microsecond.

    Raises
    ------
    ValueError
        If `seconds` is negative.
    """
    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    return round(seconds * 1_000_000)


def seconds_to_ms(seconds: float) -> int:
    """Convert seconds to milliseconds.

    Parameters
    ----------
    seconds : float
        Non-negative duration in seconds.

    Returns
    -------
    int
        Duration rounded to the nearest millisecond.

    Raises
    ------
    ValueError
        If `seconds` is negative.
    """
    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    return round(seconds * 1_000)


def ps_to_seconds(ps: int) -> float:
    """Convert picoseconds to seconds.

    Parameters
    ----------
    ps : int
        Non-negative duration in picoseconds.

    Returns
    -------
    float
        Duration in seconds.

    Raises
    ------
    ValueError
        If `ps` is negative.
    """
    if ps < 0:
        raise ValueError("picoseconds cannot be negative")
    return ps / 1_000_000_000_000


def seconds_to_ticks(seconds: float) -> SimTick:
    """Convert seconds to simulation ticks.

    Parameters
    ----------
    seconds : float
        Non-negative duration in seconds.

    Returns
    -------
    SimTick
        Integer tick count using the repository default ``1 tick == 1 ps``
        mapping.

    Raises
    ------
    ValueError
        If `seconds` is negative.

    Notes
    -----
    Conversion uses Python's :func:`round`, so fractional picoseconds follow
    Python's round-to-nearest semantics.
    """
    return SimTick(seconds_to_ps(seconds))


def ticks_to_seconds(ticks: int | SimTick) -> float:
    """Convert simulation ticks to seconds.

    Parameters
    ----------
    ticks : int or SimTick
        Non-negative simulation tick count.

    Returns
    -------
    float
        Duration in seconds using the repository default ``1 tick == 1 ps``
        mapping.

    Raises
    ------
    ValueError
        If `ticks` is negative.
    """
    return ps_to_seconds(ticks)


__all__ = [
    "DEFAULT_TICK_PS",
    "DEFAULT_TICK_SECONDS",
    "Hertz",
    "SimTick",
    "ps_to_seconds",
    "seconds_to_ms",
    "seconds_to_ns",
    "seconds_to_ps",
    "seconds_to_ticks",
    "seconds_to_us",
    "ticks_to_seconds",
]
