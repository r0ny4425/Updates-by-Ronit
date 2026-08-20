"""Test-only mock components used by simulator tests."""

from .channel_stub import ChannelStub
from .detector_stub import DetectorStub
from .emitter_stub import EmitterStub
from .signal_sink import ACTION_RECEIVE_SIGNAL, SignalSink

__all__ = [
    "ACTION_RECEIVE_SIGNAL",
    "ChannelStub",
    "DetectorStub",
    "EmitterStub",
    "SignalSink",
]
