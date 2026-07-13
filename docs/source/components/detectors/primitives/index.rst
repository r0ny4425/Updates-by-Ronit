Detector Primitives
===================

Detector primitives are the small records, policies, and helper functions used
inside detector components. They describe detector parameters, gate timing,
measurement selection, readout mapping, raw clicks, report records, and
deterministic RNG streams.

Use primitives when you want to understand or customize one part of detector
behavior.

How The Pieces Fit Together
---------------------------

A signal-facing detector component usually uses primitives in this order:

.. code-block:: text

   detector parameters
      -> measurement choice
      -> readout mapping
      -> gate and window clipping
      -> detector-channel sampling
      -> raw clicks
      -> click resolution
      -> detection report

Measurement primitives choose and execute qstate operations. Readout primitives
map qstate results to detector exposures. Window helpers apply gates and call
detector-channel models. Click resolvers turn raw clicks into user-facing
reports.

Primitive Guide
---------------

.. list-table::
   :header-rows: 1
   :widths: 26 44 30

   * - Page
     - What it explains
     - Start here when
   * - :doc:`Parameters <params>`
     - Physical channel settings such as efficiency, dark counts, jitter,
       dead time, and afterpulsing.
     - You are configuring detector channels.
   * - :doc:`Gate Windows <gate>`
     - Active detector intervals and half-open timing windows.
     - You need gated detection.
   * - :doc:`Dark Counts <dark_counts>`
     - Poisson dark-count sampling inside active windows.
     - You need background clicks.
   * - :doc:`RNG Streams <rng>`
     - Per-detector deterministic RNG stream bundles.
     - You need reproducible detector sampling.
   * - :doc:`Measurement <measurement>`
     - Measurement selection and qstate execution.
     - You need basis, POVM, Bell, random, or metadata-selected measurements.
   * - :doc:`Readout <readout>`
     - Mapping qstate result labels to detector exposures.
     - You need to map logical outcomes onto physical detector ids.
   * - :doc:`Windows <window>`
     - Gate clipping, detector ordering, RNG binding, and window evaluation.
     - You are working on detector internals.
   * - :doc:`Click Resolution <click>`
     - Turning raw clicks into logical detection reports.
     - You need double-click or click-pattern behavior.
   * - :doc:`Reports <reports>`
     - ``RawClick`` and ``DetectionReport`` records.
     - You are inspecting detector output.
   * - :doc:`Result Labels <result_labels>`
     - Extracting labels from qstate and readout result objects.
     - You need label-compatible result handling.
   * - :doc:`Actions <actions>`
     - Detector event action constants and payload expectations.
     - You are scheduling detector events.

Shared Assumptions
------------------

* Signal detection is sampled from detector efficiency.
* Dark counts are sampled from a Poisson process over the active window.
* Gates use half-open intervals, ``[start, end)``.
* Jitter is non-negative detector latency.
* Dead time blocks later accepted clicks while the channel recovers.
* Afterpulsing decays from the previous accepted click.
* Detector RNG streams are bound from the timeline before execution.

Module Pages
------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Parameters <params>
   Gate Windows <gate>
   Dark Counts <dark_counts>
   RNG Streams <rng>
   Measurement <measurement>
   Readout <readout>
   Windows <window>
   Click Resolution <click>
   Reports <reports>
   Result Labels <result_labels>
   Actions <actions>
