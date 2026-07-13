Actions
=======

Detector action constants are public event labels accepted by detector
components or scheduled internally by them. They are string labels only; payload
contracts belong to the component that handles the action.

Action Summary
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Action
     - Component
     - Payload
   * - ``ACTION_DETECT_SIGNAL``
     - ``DetectorArray``
     - ``PortDelivery`` carrying a ``Signal``.
   * - ``ACTION_RUN_QUBIT_READOUT``
     - ``QubitReadoutDevice``
     - ``QubitReadoutJob``.
   * - ``ACTION_RUN_BELL_ANALYSIS``
     - ``BellStateAnalyzer``
     - ``PortDelivery`` carrying a one-target ``Signal``.
   * - ``ACTION_COINCIDENCE_TIMEOUT``
     - ``BellStateAnalyzer``
     - Internal ``CoincidenceTimeout`` payload.
   * - ``ACTION_DARK_CANDIDATE``
     - Reserved
     - Not handled by current detector components.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/primitives/actions.py``

.. automodule:: simyuj.components.detectors.primitives.actions
   :members: ACTION_DETECT_SIGNAL, ACTION_RUN_QUBIT_READOUT, ACTION_RUN_BELL_ANALYSIS, ACTION_DARK_CANDIDATE, ACTION_COINCIDENCE_TIMEOUT
   :show-inheritance:
