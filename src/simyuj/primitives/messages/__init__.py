"""
Message records exported by the primitives package.

The package-level import path exposes the transport records most often
constructed by control and component code. Implementation modules remain split
by concern, while this package re-exports the public record classes.
"""

from .transport import ClassicalMessage, DeliveryReport, QuantumTransitPayload

__all__ = [
    "ClassicalMessage",
    "DeliveryReport",
    "QuantumTransitPayload",
]
