Coherent Optics
===============

Optical arithmetic on coherent-state amplitudes. The value type itself lives in
``simyuj.primitives.coherent_state`` and holds no math; every operation on an
amplitude lives here.

Scope
-----

Functions ship when they have a caller in ``src/``. Today that is ``attenuated``
and ``phase_shifted``, used by ``QuantumChannel``'s coherent-amplitude path, plus
``split_50_50``, ``gaussian_temporal_overlap`` and ``interfere``, used by
``DelayInterferometer``. ``click_probability``, ``polarization_weights`` and
``rotated_polarization`` arrive with the optical detector that needs them.

Both beamsplitters use the real 50:50 matrix, fixed once in the module
docstring. The symmetric convention describes the same device but puts the
interference term in the imaginary part; the two must never be mixed.
``gaussian_temporal_overlap`` takes **field**-envelope widths, matching
``Signal.temporal_mode_sigma_s``, and its ``delta_s`` is a centre-to-centre
separation -- a signal's tick is the centre of its temporal mode, so that is a
plain difference of delivery ticks.

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
