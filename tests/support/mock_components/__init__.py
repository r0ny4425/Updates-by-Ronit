"""Test-only mock components used by simulator tests."""

from .channel_stub import ChannelStub
from .detector_stub import DetectorStub
from .emitter_stub import EmitterStub

__all__ = ["ChannelStub", "DetectorStub", "EmitterStub"]
