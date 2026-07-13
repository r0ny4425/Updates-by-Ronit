from __future__ import annotations

"""Shared validation helpers for operation metadata."""


def check_arity(arity: object, *, name: str = "arity") -> int:
    """Validate a positive qubit-operation arity.

    Parameters
    ----------
    arity : object
        Candidate number of qubit operands.  The value must be exactly an
        ``int``; boolean values are rejected because ``type(value) is int`` is
        used.
    name : str, default="arity"
        Name inserted into exception messages.

    Returns
    -------
    int
        The validated positive arity.

    Raises
    ------
    TypeError
        If ``arity`` is not exactly an ``int``.
    ValueError
        If ``arity`` is zero or negative.
    """
    if type(arity) is not int:
        raise TypeError(f"{name} must be int")
    if arity <= 0:
        raise ValueError(f"{name} must be positive")
    return arity


__all__ = ["check_arity"]
