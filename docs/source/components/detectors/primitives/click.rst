Click Resolution
================

Click-resolution primitives convert low-level detector clicks into a
``DetectionReport``. They run after measurement, readout mapping, and detector
window evaluation.

Their job is to decide what the observed clicks mean.

Threshold Clicks
----------------

``ThresholdClickResolver`` is the standard resolver for detector arrays. It
uses ``RawClick.outcome_label`` to produce the logical report outcome.

.. code-block:: python

   from simyuj.components.detectors.primitives.click import (
       ThresholdClickResolver,
   )

   resolver = ThresholdClickResolver(double_click_policy="fail")

With no clicks, the report fails with ``FLAG_NO_CLICK``. With one click, the
report succeeds and uses that click's ``outcome_label``.

Where These Are Used
--------------------

``ThresholdClickResolver`` is the usual resolver for ``DetectorArray`` because
readout has already copied a logical outcome label onto each raw click.
``ClickPattern`` is more common in Bell-state and coincidence-style models,
where an exact clicked-detector tuple maps to a logical outcome.

Double Clicks
-------------

A double click means more than one raw click was available for the same report.
This can happen when two signal channels fire, or when a dark count appears in
another detector during the same detection window.

``ThresholdClickResolver`` supports three policies:

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Policy
     - Behavior
     - Use when
   * - ``"fail"``
     - Produce a failed report with ``FLAG_DOUBLE_CLICK``.
     - You want conservative protocol filtering.
   * - ``"first"``
     - Choose the earliest sorted click.
     - The earliest firing detector should win.
   * - ``"random"``
     - Choose uniformly from the sorted clicks using the resolver RNG.
     - Double clicks should be randomly assigned.

Raw clicks are sorted by click time, detector id, and trigger before
resolution.

Click Patterns
--------------

``ClickPattern`` maps an exact set of clicked detector ids to a logical
outcome. This is useful for Bell-state analyzer models and coincidence-style
detection.

.. code-block:: python

   from simyuj.components.detectors.primitives.click import (
       ClickPattern,
       resolve_click_pattern,
   )
   from simyuj.components.detectors.primitives.reports import RawClick

   patterns = (
       ClickPattern(outcome="psi+", detector_ids=("d0", "d3")),
       ClickPattern(outcome="psi-", detector_ids=("d1", "d2")),
   )

   outcome = resolve_click_pattern(
       raw_clicks=(
           RawClick(detector_id="d0", time=10, trigger="signal"),
           RawClick(detector_id="d3", time=10, trigger="signal"),
       ),
       patterns=patterns,
   )

Pattern matching is exact after sorting detector ids.

POVM Result Labels
------------------

``POVMLabelClickResolver`` uses the qstate measurement result label as the
reported outcome. This is useful when the POVM outcome itself is the detector
result and physical click labels should not replace it.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/primitives/click.py``

.. automodule:: simyuj.components.detectors.primitives.click
   :members: ClickPatternResolver, ClickPattern, resolve_click_pattern, ThresholdClickResolver, POVMLabelClickResolver
   :show-inheritance:
