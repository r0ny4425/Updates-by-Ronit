Result Labels
=============

``result_label(...)`` extracts a label-like value from qstate or readout result
objects without depending on one concrete result class.

The helper checks ``label``, then ``outcome_label``, then ``outcome``. If none
exist, it returns the original object. ``None`` stays ``None``.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/primitives/result_labels.py``

.. automodule:: simyuj.components.detectors.primitives.result_labels
   :members: result_label
   :show-inheritance:
