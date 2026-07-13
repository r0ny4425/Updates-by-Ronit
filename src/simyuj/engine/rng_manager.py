"""Deterministic random-number streams owned by the timeline.

The engine routes stochastic behavior through named streams derived from one
master seed. Stream identity depends on the hierarchical path, not on the order
in which streams are requested, so fixed seeds and fixed stream paths reproduce
the same draw sequences.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np


def _path_to_spawn_key(parts: Iterable[str]) -> Tuple[int, ...]:
    """Convert a stream path into a deterministic NumPy spawn key.

    Parameters
    ----------
    parts : Iterable[str]
        Normalized stream path segments.

    Returns
    -------
    Tuple[int, ...]
        Four unsigned 32-bit integers derived from the first 128 bits of a
        SHA-256 digest.

    Notes
    -----
    Python's built-in ``hash`` is intentionally randomized between processes,
    so the manager uses SHA-256 for reproducible path-derived entropy.
    """
    h = hashlib.sha256("/".join(parts).encode("utf-8")).digest()
    return tuple(
        int.from_bytes(h[i : i + 4], "big", signed=False) for i in range(0, 16, 4)
    )


class DeterministicRNG:
    """Restricted wrapper around ``numpy.random.Generator``.

    Components receive this wrapper instead of the raw NumPy generator. The
    wrapper exposes approved draw methods while blocking attribute mutation,
    reseeding, and direct generator-state access.

    Parameters
    ----------
    gen : numpy.random.Generator
        Generator created by ``RNGManager`` for one stream path.

    Notes
    -----
    The wrapper is attribute-immutable, but draw methods still advance the
    private generator state.
    """

    __slots__ = ("__gen",)

    def __init__(self, gen: np.random.Generator):
        object.__setattr__(self, "_DeterministicRNG__gen", gen)

    def random(self) -> float:
        """Draw a float uniformly from ``[0.0, 1.0)``.

        Returns
        -------
        float
            Uniform random draw.
        """
        return float(self.__gen.random())

    def randint(self, a: int, b: int) -> int:
        """Draw an integer uniformly from the inclusive range ``[a, b]``.

        Parameters
        ----------
        a : int
            Inclusive lower bound.
        b : int
            Inclusive upper bound.

        Returns
        -------
        int
            Random integer in ``[a, b]``.

        Notes
        -----
        Bounds are forwarded to NumPy after converting the upper bound to
        NumPy's exclusive form. The method does not add separate type or range
        validation.
        """
        return int(self.__gen.integers(a, b + 1))

    def uniform(self, a: float, b: float) -> float:
        """Draw a float uniformly from ``[a, b)``.

        Parameters
        ----------
        a : float
            Lower bound.
        b : float
            Exclusive upper bound.

        Returns
        -------
        float
            Uniform random draw.
        """
        return float(self.__gen.uniform(a, b))

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        """Draw a float from a normal distribution.

        Parameters
        ----------
        loc : float, default=0.0
            Mean of the distribution.
        scale : float, default=1.0
            Standard deviation passed to NumPy.

        Returns
        -------
        float
            Normal random draw.
        """
        return float(self.__gen.normal(loc, scale))

    def poisson(self, lam: float) -> int:
        """Draw an integer from a Poisson distribution.

        Parameters
        ----------
        lam : float
            Poisson rate. Must be non-negative.

        Returns
        -------
        int
            Poisson random draw.

        Raises
        ------
        ValueError
            If ``lam`` is negative.
        """
        if lam < 0:
            raise ValueError("Poisson λ must be non-negative")
        return int(self.__gen.poisson(lam))

    def exponential(self, scale: float) -> float:
        """Draw a float from an exponential distribution.

        Parameters
        ----------
        scale : float
            Distribution scale. Must be positive.

        Returns
        -------
        float
            Exponential random draw.

        Raises
        ------
        ValueError
            If ``scale`` is not positive.
        """
        if scale <= 0:
            raise ValueError("Exponential scale must be positive")
        return float(self.__gen.exponential(scale))

    def choice(self, seq: Sequence) -> Any:
        """Draw one element from an indexable sequence.

        Parameters
        ----------
        seq : Sequence
            Indexable sequence such as a list, tuple, string, or NumPy array.

        Returns
        -------
        Any
            Element selected from ``seq``.

        Raises
        ------
        TypeError
            If ``seq`` does not provide ``__len__`` and ``__getitem__``.
        IndexError
            If ``seq`` is empty.
        """
        if not hasattr(seq, "__len__") or not hasattr(seq, "__getitem__"):
            raise TypeError(
                "choice() requires an indexable sequence (list, tuple, etc.), "
                "not a generator or iterator"
            )
        if len(seq) == 0:
            raise IndexError("Cannot choose from empty sequence")
        idx = int(self.__gen.integers(0, len(seq)))
        return seq[idx]

    def __getattr__(self, name: str) -> Any:
        """Block access to raw generator attributes and unsupported methods."""
        raise AttributeError(
            f"Access to '{name}' is forbidden on DeterministicRNG. "
            f"Only approved draw methods are available."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Block external mutation of wrapper attributes."""
        raise AttributeError(
            f"DeterministicRNG is immutable; Cannot set attribute '{name}'. "
            f"This object is immutable."
        )

    def __repr__(self) -> str:
        return "DeterministicRNG()"


class RNGManager:
    """Timeline-owned manager for deterministic RNG streams.

    ``RNGManager`` creates one restricted ``DeterministicRNG`` per normalized
    path. Paths are derived through NumPy ``SeedSequence`` with PCG64, so the
    same master seed and path produce the same stream across independent
    managers, regardless of stream creation order.

    Parameters
    ----------
    master_seed : int
        Non-negative integer used as the root seed.
    owner_id : str
        Identifier required by ``rng(..., requester_id=...)``. ``Timeline`` uses
        this to keep stream creation timeline-owned.

    Raises
    ------
    ValueError
        If ``master_seed`` is not a non-negative integer.

    Notes
    -----
    ``freeze`` prevents creation of new streams after execution begins, while
    existing stream objects remain usable and keep advancing their draw state.
    The manager uses a lock around stream lookup and creation.
    """

    # --- mypy-visible attribute declarations ---
    _root_seed_seq: np.random.SeedSequence
    _streams: dict[Tuple[str, ...], DeterministicRNG]
    _owner_id: str
    _frozen: bool
    _lock: threading.Lock

    __slots__ = (
        "_root_seed_seq",
        "_streams",
        "_owner_id",
        "_frozen",
        "_lock",
    )

    def __init__(self, master_seed: int, owner_id: str):
        if not isinstance(master_seed, int) or master_seed < 0:
            raise ValueError("master_seed must be a non-negative int")

        object.__setattr__(
            self,
            "_root_seed_seq",
            np.random.SeedSequence(master_seed),
        )
        object.__setattr__(self, "_streams", {})
        object.__setattr__(self, "_frozen", False)
        object.__setattr__(self, "_owner_id", owner_id)
        object.__setattr__(self, "_lock", threading.Lock())

    def rng(
        self,
        *path: str,
        requester_id: Optional[str] = None,
    ) -> DeterministicRNG:
        """Return the deterministic stream for a hierarchical path.

        Parameters
        ----------
        *path : str
            Hierarchical stream path. Segments are converted to strings and
            stripped; empty segments and slash characters are rejected.
        requester_id : str, optional
            Identifier that must match the manager owner.

        Returns
        -------
        DeterministicRNG
            Cached stream object for ``path``.

        Raises
        ------
        PermissionError
            If ``requester_id`` does not match ``owner_id``.
        RuntimeError
            If a new stream is requested after ``freeze``.
        ValueError
            If the path is empty or contains an invalid segment.

        Examples
        --------
        >>> mgr = RNGManager(master_seed=7, owner_id="timeline")
        >>> rng = mgr.rng("alice", "basis", requester_id="timeline")
        >>> round(rng.random(), 6)
        0.689974
        """
        if requester_id != self._owner_id:
            raise PermissionError(
                f"Only Timeline '{self._owner_id}' may request RNG streams, "
                f"but got requester_id='{requester_id}'"
            )

        path = self._normalize_path(path)

        # Thread-safe stream creation
        with self._lock:
            # Check if stream already exists
            if path in self._streams:
                return self._streams[path]

            # Check if frozen
            if self._frozen:
                raise RuntimeError(
                    "RNG stream requested after freeze(): " f"{'/'.join(path)}"
                )

            # Create new stream
            seed_seq = np.random.SeedSequence(
                entropy=self._root_seed_seq.entropy,
                spawn_key=_path_to_spawn_key(path),
            )
            gen = np.random.Generator(np.random.PCG64(seed_seq))
            drng = DeterministicRNG(gen)

            self._streams[path] = drng

            return drng

    def freeze(self) -> None:
        """Prevent creation of streams that have not already been declared.

        Existing streams can still be requested by the same path and can still
        generate random numbers. New paths raise ``RuntimeError`` after the
        manager is frozen.

        Notes
        -----
        ``Timeline`` freezes the manager when execution begins so stochastic
        sources must be declared before the first batch is run.
        """
        with self._lock:
            object.__setattr__(self, "_frozen", True)

    def list_streams(self) -> Tuple[Tuple[str, ...], ...]:
        """Return created stream paths in deterministic sorted order.

        Returns
        -------
        Tuple[Tuple[str, ...], ...]
            Normalized stream paths.
        """
        with self._lock:
            return tuple(sorted(self._streams.keys()))

    def stream_count(self) -> int:
        """Return the number of created streams."""
        with self._lock:
            return len(self._streams)

    def has_stream(self, *path: str) -> bool:
        """Return whether a normalized stream path already exists.

        Parameters
        ----------
        *path : str
            Stream path checked with the same normalization rules as ``rng``.

        Returns
        -------
        bool
            ``True`` if the stream has already been created.

        Raises
        ------
        ValueError
            If the path is empty or contains an invalid segment.
        """
        normalized = self._normalize_path(path)
        with self._lock:
            return normalized in self._streams

    def _normalize_path(self, path: Tuple[str, ...]) -> Tuple[str, ...]:
        """Normalize path segments and reject ambiguous separators."""
        if not path:
            raise ValueError("RNG stream path must not be empty")

        parts = []
        for p in path:
            p = str(p).strip()
            if not p:
                raise ValueError("RNG path segment(s) must be non-empty")
            if "/" in p or "\\" in p:
                raise ValueError(
                    f"Invalid RNG path segment '{p}': cannot contain / or \\ characters"
                )
            parts.append(p)

        return tuple(parts)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent external modification of RNGManager state."""
        raise AttributeError(
            f"Cannot set attribute '{name}' on RNGManager. "
            f"RNGManager state is immutable after construction."
        )

    def __repr__(self) -> str:
        return f"RNGManager(streams={len(self._streams)}, " f"frozen={self._frozen})"
