import pytest

from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.engine.timeline_statistics import TimelineStatistics

# Stats Do Not Affect Execution


def test_reading_stats_during_execution():
    """Reading stats during execution should not affect behavior."""
    timeline = Timeline(master_seed=42)

    execution_order = []

    class Handler:
        def __init__(self, name):
            self.name = name

        def handle_event(self, event, timeline):
            _ = timeline.stats
            _ = timeline.events_executed
            execution_order.append(self.name)

    h1 = Handler("A")
    h2 = Handler("B")
    h3 = Handler("C")

    timeline.schedule(Event(time=10, target_ref=h1, action="TEST", payload_ref=None))
    timeline.schedule(Event(time=20, target_ref=h2, action="TEST", payload_ref=None))
    timeline.schedule(Event(time=30, target_ref=h3, action="TEST", payload_ref=None))

    timeline.run_until(50)

    assert execution_order == ["A", "B", "C"]


def test_stats_snapshot_is_immutable():
    """TimelineStatistics should be immutable (frozen dataclass)."""
    timeline = Timeline(master_seed=42)

    stats = timeline.stats

    with pytest.raises(Exception):
        stats.total_events_scheduled = 999

    with pytest.raises(Exception):
        stats.total_events_executed = 999


#  Identical Runs Produce Identical Statistics


def test_same_seed_same_stats():
    """Same seed and same events should produce identical statistics."""

    def run_simulation(seed: int) -> TimelineStatistics:
        timeline = Timeline(master_seed=seed)

        class Handler:
            def handle_event(self, event, timeline):
                pass

        handler = Handler()

        # Schedule 20 events
        for i in range(20):
            timeline.schedule(
                Event(time=i * 5, target_ref=handler, action="TEST", payload_ref=None)
            )

        timeline.run_until(100)

        return timeline.stats

    stats1 = run_simulation(42)
    stats2 = run_simulation(42)
    stats3 = run_simulation(42)

    assert stats1 == stats2 == stats3


def test_different_seeds_may_differ():
    """Different seeds may produce different statistics if RNG affects scheduling."""

    def run_with_random_scheduling(seed: int) -> TimelineStatistics:
        timeline = Timeline(master_seed=seed)
        rng = timeline.rng("scheduler")

        class Handler:
            def handle_event(self, event, timeline):
                pass

        handler = Handler()

        # Schedule random number of events
        n_events = rng.randint(10, 20)
        for i in range(n_events):
            timeline.schedule(
                Event(time=i * 10, target_ref=handler, action="TEST", payload_ref=None)
            )

        timeline.run_until(200)
        return timeline.stats

    stats1 = run_with_random_scheduling(42)
    stats2 = run_with_random_scheduling(99)

    assert run_with_random_scheduling(42) == stats1
    assert run_with_random_scheduling(99) == stats2


def test_stats_deterministic_with_cancellation():
    """Statistics should be deterministic even with event cancellation."""

    def run_with_cancellation(seed: int) -> TimelineStatistics:
        timeline = Timeline(master_seed=seed)

        class Handler:
            def handle_event(self, event, timeline):
                pass

        handler = Handler()

        # Schedule 10 events
        events = []
        for i in range(10):
            e = timeline.schedule(
                Event(time=i * 10, target_ref=handler, action="TEST", payload_ref=None)
            )
            events.append(e)

        # Cancel every other event
        for i in range(0, 10, 2):
            timeline.cancel(events[i])

        timeline.run_until(100)
        return timeline.stats

    stats1 = run_with_cancellation(12345)
    stats2 = run_with_cancellation(12345)

    assert stats1 == stats2

    # Should have scheduled 10 but executed only 5
    assert stats1.total_events_scheduled == 10
    assert stats1.total_events_executed == 5


#  Stats Increment Correctly During Execution


def test_events_executed_with_batching():
    """events_executed should count all events in a batch."""
    timeline = Timeline(master_seed=42)

    class Handler:
        def handle_event(self, event, timeline):
            pass

    handler = Handler()

    # Schedule 5 events at the SAME time (will execute as batch)
    for i in range(5):
        timeline.schedule(
            Event(
                time=100,
                priority=i,
                target_ref=handler,
                action="TEST",
                payload_ref=None,
            )
        )

    assert timeline.events_executed == 0

    timeline.run_one_step()

    # Should have executed all 5
    assert timeline.events_executed == 5


def test_max_queue_size_tracks_maximum():
    """max_queue_size should track the maximum size reached."""
    timeline = Timeline(master_seed=42)

    class Handler:
        def handle_event(self, event, timeline):
            pass

    handler = Handler()

    assert timeline.max_queue_size == 0

    # Add 1 event
    timeline.schedule(
        Event(time=10, target_ref=handler, action="TEST", payload_ref=None)
    )
    assert timeline.max_queue_size == 1

    # Add 2 more (total 3)
    timeline.schedule(
        Event(time=20, target_ref=handler, action="TEST", payload_ref=None)
    )
    timeline.schedule(
        Event(time=30, target_ref=handler, action="TEST", payload_ref=None)
    )
    assert timeline.max_queue_size == 3

    # Execute one (queue now has 2)
    timeline.run_one_step()
    # max_queue_size should still be 3 (maximum ever reached)
    assert timeline.max_queue_size == 3

    # Add more events (total queue size goes to 5)
    timeline.schedule(
        Event(time=40, target_ref=handler, action="TEST", payload_ref=None)
    )
    timeline.schedule(
        Event(time=50, target_ref=handler, action="TEST", payload_ref=None)
    )
    timeline.schedule(
        Event(time=60, target_ref=handler, action="TEST", payload_ref=None)
    )
    # Now max should be updated
    assert timeline.max_queue_size == 5


def test_cancelled_events_affect_scheduled_not_executed():
    """Cancelled events count as scheduled but not executed."""
    timeline = Timeline(master_seed=42)

    class Handler:
        def handle_event(self, event, timeline):
            pass

    handler = Handler()

    # Schedule 5 events
    events = []
    for i in range(5):
        e = timeline.schedule(
            Event(time=i * 10, target_ref=handler, action="TEST", payload_ref=None)
        )
        events.append(e)

    assert timeline.events_scheduled == 5

    # Cancel 2 events
    timeline.cancel(events[1])
    timeline.cancel(events[3])

    # Still scheduled (cancellation doesn't decrement)
    assert timeline.events_scheduled == 5

    # Execute all
    timeline.run_until(100)

    # Only 3 should have executed (2 were cancelled)
    assert timeline.events_executed == 3


# Reading Stats Has No Side Effects


def test_stats_read_doesnt_affect_scheduling():
    """Reading stats should not affect ability to schedule events."""
    timeline = Timeline(master_seed=42)

    class Handler:
        def handle_event(self, event, timeline):
            pass

    handler = Handler()

    # Read stats
    _ = timeline.stats

    # Should still be able to schedule
    timeline.schedule(
        Event(time=10, target_ref=handler, action="TEST", payload_ref=None)
    )

    assert timeline.events_scheduled == 1


def test_stats_read_doesnt_affect_execution():
    """Reading stats should not affect execution."""
    timeline = Timeline(master_seed=42)

    executed = []

    class Handler:
        def __init__(self, name):
            self.name = name

        def handle_event(self, event, timeline):
            executed.append(self.name)
            # Read stats during execution
            _ = timeline.stats

    h1 = Handler("A")
    h2 = Handler("B")

    timeline.schedule(Event(time=10, target_ref=h1, action="TEST", payload_ref=None))
    timeline.schedule(Event(time=20, target_ref=h2, action="TEST", payload_ref=None))

    timeline.run_until(30)

    assert executed == ["A", "B"]


def test_multiple_stats_reads_same_result():
    """Multiple reads of stats should return same values."""
    timeline = Timeline(master_seed=42)

    class Handler:
        def handle_event(self, event, timeline):
            pass

    handler = Handler()

    timeline.schedule(
        Event(time=10, target_ref=handler, action="TEST", payload_ref=None)
    )
    timeline.run_one_step()

    # Read 100 times
    stats_list = [timeline.stats for _ in range(100)]

    # All should be identical
    for stats in stats_list:
        assert stats == stats_list[0]


# EDGE CASES


def test_stats_with_no_events():
    """Statistics should be valid even with no events."""
    timeline = Timeline(master_seed=42)

    stats = timeline.stats
    assert stats.total_events_scheduled == 0
    assert stats.total_events_executed == 0
    assert stats.max_queue_size == 0
    assert stats.current_time == 0


def test_stats_with_only_scheduling_no_execution():
    """Stats should be correct if events are scheduled but not executed."""
    timeline = Timeline(master_seed=42)

    class Handler:
        def handle_event(self, event, timeline):
            pass

    handler = Handler()

    # Schedule but don't execute
    for i in range(10):
        timeline.schedule(
            Event(time=i * 10, target_ref=handler, action="TEST", payload_ref=None)
        )

    stats = timeline.stats
    assert stats.total_events_scheduled == 10
    assert stats.total_events_executed == 0
    assert stats.max_queue_size == 10


def test_stats_with_very_large_numbers():
    """Statistics should handle large numbers correctly."""
    timeline = Timeline(master_seed=42)

    class Handler:
        def handle_event(self, event, timeline):
            pass

    handler = Handler()

    # Schedule many events
    for i in range(10000):
        timeline.schedule(
            Event(time=i, target_ref=handler, action="TEST", payload_ref=None)
        )

    stats = timeline.stats
    assert stats.total_events_scheduled == 10000
    assert stats.max_queue_size == 10000
