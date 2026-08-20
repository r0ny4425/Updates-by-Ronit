"""Tests for the coherent-optics feature: the amplitude value type, the
optical fields on ``Signal``, and the port-based sink they are exercised with.

The directory is a package so tests here can import shared scaffolding from
``tests.support``. ``tests/signal/`` predates that convention and has no
``__init__.py``, which is why a relative import fails from inside it.
"""
