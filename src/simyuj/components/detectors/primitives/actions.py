"""Canonical detector event action names.

The constants in this module are the public event actions accepted by detector
components or scheduled internally by those components. They are string labels
only; payload contracts are documented on the component or record that handles
the action.
"""

from __future__ import annotations

ACTION_DETECT_SIGNAL = "detect_signal"
"""Quantum-signal detection action accepted by ``DetectorArray``.

The event payload must be a ``PortDelivery`` whose payload is a ``Signal`` and
whose target port is the array input port.
"""

ACTION_RUN_QUBIT_READOUT = "run_qubit_readout"
"""Explicit qstate readout action accepted by ``QubitReadoutDevice``.

The event payload must be a ``QubitReadoutJob``.
"""

ACTION_RUN_BELL_ANALYSIS = "run_bell_analysis"
"""Quantum input action accepted by ``BellStateAnalyzer`` left/right ports.

The event payload must be a ``PortDelivery`` carrying a one-target ``Signal``.
"""

ACTION_DARK_CANDIDATE = "dark_candidate"
"""Reserved detector dark-count candidate action.

The current detector implementation samples dark counts while evaluating a
detection window instead of handling this action directly. The action name is
kept for future event-scheduled dark-count models.
"""

ACTION_COINCIDENCE_TIMEOUT = "coincidence_timeout"
"""Internal Bell-analyzer timeout action for unmatched buffered inputs.

The event payload must be a ``CoincidenceTimeout`` created by the analyzer when
one side arrives without a coincident partner.
"""


__all__ = [
    "ACTION_COINCIDENCE_TIMEOUT",
    "ACTION_DARK_CANDIDATE",
    "ACTION_DETECT_SIGNAL",
    "ACTION_RUN_BELL_ANALYSIS",
    "ACTION_RUN_QUBIT_READOUT",
]
