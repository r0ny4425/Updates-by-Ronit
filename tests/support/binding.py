from __future__ import annotations

from simyuj.engine.timeline import Timeline
from simyuj.runtime.binding import BindingContext


def binding_context(timeline: Timeline) -> BindingContext:
    return BindingContext(timeline=timeline, logger=timeline.logger)
