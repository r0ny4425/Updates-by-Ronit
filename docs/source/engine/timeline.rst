Timeline
========

Qstate Noise Mode
-----------------

``Timeline`` accepts ``qstate_noise_mode`` and passes it to the timeline-owned
``QuantumStateManager``. The default ``"density"`` mode keeps exact noise
evolution. ``"sampled_ket"`` predeclares the deterministic ``qstate/noise`` RNG
stream and lets qstate noise-model chains sample ket trajectories without
changing component APIs.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/engine/timeline.py``

.. automodule:: simyuj.engine.timeline
   :members:
   :show-inheritance:
