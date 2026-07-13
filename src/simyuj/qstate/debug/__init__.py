from __future__ import annotations

"""Debugging helpers for inspecting qstate records and store invariants.

This package is intended for tests, diagnostics, and interactive development.
Dump helpers return strings rather than printing, while invariant helpers raise
domain errors when records or store indexes are inconsistent.
"""

from .dump import dump_layout, dump_payload_summary, dump_record, dump_store
from .invariant import (
    assert_layout_matches_payload,
    assert_record_ok,
    assert_store_ok,
    assert_unique_subsystems,
    assert_valid_state,
    iter_store_records,
)

# Public debug helpers re-exported from ``simyuj.qstate.debug``.
__all__ = [
    "assert_layout_matches_payload",
    "assert_record_ok",
    "assert_store_ok",
    "assert_unique_subsystems",
    "assert_valid_state",
    "dump_layout",
    "dump_payload_summary",
    "dump_record",
    "dump_store",
    "iter_store_records",
]
