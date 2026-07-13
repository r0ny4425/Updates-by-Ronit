from __future__ import annotations

"""Finite classical samplers for qstate preparation payloads.

This public module provides ``StateSampler`` and ``StateSample`` for workflows
that choose one quantum-state initializer from a finite classical distribution.
The sampler builds payloads from candidates through the selected qstate
representation handler at construction time, but it does not own subsystems,
allocate state references, or mutate a ``QuantumStateStore``.

Random draws require an explicit caller-supplied RNG stream. Timeline-driven
simulations should pass a stream obtained from ``Timeline.rng(...)`` so state
selection participates in deterministic replay.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .check import normalize_rep
from .errors import InvalidReprError
from .math.const import PROB_ATOL
from .math.prob import check_prob_vector
from .rng import sample_index
from .state import BellDiagHandler, DensityHandler, KetHandler
from .state.check import _payload_num_qubits_trusted


@dataclass(frozen=True, slots=True)
class StateSample:
    """One selected qstate payload and its sampling metadata.

    Attributes
    ----------
    state : object
        Qstate payload selected from ``StateSampler.states``.
    rep : str
        Normalized representation name associated with ``state``. Current
        samplers support ``"ket"``, ``"density"``, and ``"bell_diag"``.
    index : int
        Position of the selected payload in the sampler. Use this index to look
        up the corresponding entry in ``StateSampler.probabilities``.
    label : str or None, optional
        Optional user-supplied classical label for the selected payload.
    """

    state: object
    rep: str
    index: int
    label: str | None = None


@dataclass(frozen=True, slots=True)
class StateSampler:
    """Finite sampler for qstate payload preparations.

    The sampler represents a classical distribution over candidate qstate
    payloads. Candidate states are built once during construction using the
    representation handler selected by ``rep``. The stored ``states``,
    ``probabilities``, and ``labels`` attributes are immutable tuples after
    construction.

    Parameters
    ----------
    states : sequence of object
        Non-empty state initializers accepted by the selected representation.
        For ``rep="ket"``, entries may be predefined labels such as ``"|0>"``
        and ``"|+>"``, existing ``KetState`` payloads, or normalized complex
        vectors of shape ``(2**n,)``. For ``rep="density"``, entries may be
        density matrices of shape ``(2**n, 2**n)`` or ket initializers that can
        be converted to density payloads. For ``rep="bell_diag"``, entries may
        be Bell labels, label-to-probability mappings, four-entry probability
        vectors, or ``BellDiagState`` payloads.
    probabilities : sequence of float or None, optional
        Unit-sum classical probabilities aligned with ``states``. If omitted,
        all states are sampled uniformly.
    rep : str, default="ket"
        Qstate representation used to build each state. Aliases accepted by
        ``normalize_rep`` are allowed, but the normalized representation must be
        ``"ket"``, ``"density"``, or ``"bell_diag"``.
    labels : sequence of str or None or None, optional
        Optional classical metadata labels aligned with ``states``.

    Attributes
    ----------
    states : tuple of object
        Stored qstate payloads. All payloads must have the same qubit count.
    probabilities : tuple of float
        Checked unit-sum classical probability vector.
    rep : str
        Normalized representation name.
    labels : tuple of str or None
        Checked label tuple aligned with ``states``.

    Raises
    ------
    TypeError
        If ``states``, ``probabilities``, or ``labels`` are not supported
        sequence inputs, or if a representation-specific state constructor
        rejects an initializer type.
    ValueError
        If ``states`` is empty, if probabilities or labels do not match the
        state count, if probabilities are invalid, if labels are empty strings,
        or if candidate payloads do not all have the same qubit count.
    InvalidReprError
        If ``rep`` is unsupported or names a representation that is not handled
        by ``StateSampler``.

    Notes
    -----
    Selection probabilities are classical weights over candidate preparations;
    they are not amplitudes and are not stored on ``StateSample``. Use
    ``sample.index`` to recover the selected probability from
    ``sampler.probabilities``.

    The sampler validates and builds each candidate through the qstate
    representation layer. Numerical conventions such as computational-basis
    ordering, ket normalization, density-matrix Hermiticity, and Bell-diagonal
    label order are inherited from the selected representation handler.

    Examples
    --------
    >>> import numpy as np
    >>> class FixedRNG:
    ...     def random(self):
    ...         return 0.7
    >>> sampler = StateSampler(
    ...     states=("|0>", "|+>", np.array([0.6, 0.8], dtype=np.complex128)),
    ...     probabilities=(0.25, 0.25, 0.5),
    ...     labels=("zero", "plus", "custom"),
    ... )
    >>> sample = sampler.sample(rng=FixedRNG())
    >>> sample.index, sample.label
    (2, 'custom')
    >>> sampler.probabilities[sample.index]
    0.5
    """

    states: Sequence[object]
    probabilities: Sequence[float] | None = None
    rep: str = "ket"
    labels: Sequence[str | None] | None = None

    def __post_init__(self) -> None:
        rep = _concrete_rep(self.rep)
        states = _check_sequence(self.states, name="states")
        if not states:
            raise ValueError("states must be non-empty")

        payloads = tuple(_make_state(state, rep) for state in states)
        _check_same_qubit_count(payloads)

        object.__setattr__(self, "states", payloads)
        object.__setattr__(
            self,
            "probabilities",
            _check_probabilities(
                self.probabilities,
                state_count=len(payloads),
            ),
        )
        object.__setattr__(self, "rep", rep)
        object.__setattr__(
            self,
            "labels",
            _check_labels(
                self.labels,
                state_count=len(payloads),
            ),
        )

    @property
    def size(self) -> int:
        """Return the number of candidate states.

        Returns
        -------
        int
            Length of ``states``, ``probabilities``, and ``labels``.
        """
        return len(self.states)

    @property
    def num_qubits(self) -> int:
        """Return the common qubit count for all candidate states.

        Returns
        -------
        int
            Number of qubits reported by the stored qstate payloads.
        """
        return _payload_num_qubits_trusted(self.states[0])

    def sample(self, *, rng: Any | None = None) -> StateSample:
        """Sample one candidate payload.

        Parameters
        ----------
        rng : object or None, optional
            Explicit random-number stream used for stochastic distributions. It
            must provide ``random()`` or ``rand()`` when a draw is required.

        Returns
        -------
        StateSample
            Selected payload, normalized representation name, selected index,
            and optional label.

        Raises
        ------
        ValueError
            If the distribution is stochastic and ``rng`` is ``None``.
        TypeError
            If ``rng`` cannot produce a scalar draw when sampling is required.

        Notes
        -----
        If any probability is within ``PROB_ATOL`` of one, the first such entry
        is returned without drawing from ``rng``. Otherwise sampling delegates
        to ``qstate.rng.sample_index``, which requires normalized
        probabilities and an explicit RNG stream.
        """
        probabilities = self.probabilities
        labels = self.labels
        assert probabilities is not None
        assert labels is not None

        index = _sample_index(tuple(probabilities), rng=rng)
        return StateSample(
            state=self.states[index],
            rep=self.rep,
            index=index,
            label=labels[index],
        )


def _concrete_rep(rep: object) -> str:
    normalized = normalize_rep(rep)
    if normalized not in {"ket", "density", "bell_diag"}:
        raise InvalidReprError(
            "StateSampler supports ket, density, and bell_diag representations"
        )
    return normalized


def _check_sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


def _make_state(state: object, rep: str) -> object:
    if rep == "ket":
        return KetHandler().make(state)
    if rep == "density":
        return DensityHandler().make(state)
    if rep == "bell_diag":
        return BellDiagHandler().make(state)
    raise InvalidReprError(f"unsupported state representation: {rep!r}")


def _check_same_qubit_count(states: tuple[object, ...]) -> None:
    expected = _payload_num_qubits_trusted(states[0])
    for state in states[1:]:
        if _payload_num_qubits_trusted(state) != expected:
            raise ValueError("states must have the same qubit count")


def _check_probabilities(
    probabilities: Sequence[float] | None,
    *,
    state_count: int,
) -> tuple[float, ...]:
    if probabilities is None:
        probability = 1.0 / state_count
        return tuple(probability for _ in range(state_count))

    checked = check_prob_vector(probabilities, name="probabilities")
    if len(checked) != state_count:
        raise ValueError("probabilities length must match states length")
    return checked


def _check_labels(
    labels: Sequence[str | None] | None,
    *,
    state_count: int,
) -> tuple[str | None, ...]:
    if labels is None:
        return tuple(None for _ in range(state_count))

    checked = _check_sequence(labels, name="labels")
    if len(checked) != state_count:
        raise ValueError("labels length must match states length")
    result: list[str | None] = []
    for label in checked:
        if label is None:
            result.append(None)
            continue
        if not isinstance(label, str) or not label.strip():
            raise ValueError("labels entries must be non-empty str or None")
        result.append(label)
    return tuple(result)


def _sample_index(probabilities: tuple[float, ...], *, rng: Any | None) -> int:
    for index, probability in enumerate(probabilities):
        if abs(probability - 1.0) <= PROB_ATOL:
            return index
    return sample_index(rng, probabilities)


__all__ = ["StateSample", "StateSampler"]
