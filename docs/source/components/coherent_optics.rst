Coherent Optics
===============

Optical arithmetic on coherent-state amplitudes. The value type itself lives in
``simyuj.primitives.coherent_state`` and holds no math; every operation on an
amplitude lives here.

Scope
-----

Functions ship when they have a caller in ``src/``. Today that is ``attenuated``
and ``phase_shifted``, both used by ``QuantumChannel``'s coherent-amplitude
path. ``split_50_50``, ``interfere``, ``gaussian_temporal_overlap``,
``click_probability``, ``polarization_weights`` and ``rotated_polarization``
arrive with the receiver components that need them.

This module takes no RNG and returns no random value. Nothing here samples a
photon number; photon statistics are integrated in closed form at detection.

Power, not amplitude
--------------------

``attenuated`` takes a **power** transmission :math:`\eta`, so the amplitude
scales as :math:`\sqrt{\eta}` and the mean photon number as :math:`\eta`. It is
the same :math:`10^{-L/10}` that is a Bernoulli survival probability for a
single photon -- one fibre property with two correct consequences for two
different input states.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/coherent_optics.py``

.. automodule:: simyuj.components.coherent_optics
   :members:
   :show-inheritance:
