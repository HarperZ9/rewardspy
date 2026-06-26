import rewardspy
from rewardspy.detectors import (
    CeilingRateDetector,
    ComponentDominanceDetector,
    DetectionEngine,
    LengthDriftDetector,
    RewardSlopeChangeDetector,
    Status,
    VarianceCollapseDetector,
)
from rewardspy.records import RolloutRecord
from rewardspy.store import MetricStore


def make_record(step, reward, components=None, output_length=10):
    return RolloutRecord(
        call_id=f"c{step}",
        timestamp=float(step),
        step=step,
        scalar_reward=reward,
        components=components or {},
        output_length=output_length,
    )


def feed(store, rewards, components_seq=None, lengths=None):
    for i, r in enumerate(rewards):
        comps = components_seq[i] if components_seq else None
        length = lengths[i] if lengths else 10
        store.append(make_record(i, r, comps, length))


# -- variance ------------------------------------------------------------------

def test_variance_collapse_fires():
    store = MetricStore("t", window_size=50)
    noisy = [(i % 7) * 0.3 for i in range(60)]  # real spread early
    collapsed = [1.0] * 100  # then everything identical
    feed(store, noisy + collapsed)
    result = VarianceCollapseDetector("medium").check(store)
    assert result.status == Status.ALERT


def test_variance_healthy_is_ok():
    store = MetricStore("t", window_size=50)
    rewards = [(i % 7) * 0.3 for i in range(200)]  # stable spread throughout
    feed(store, rewards)
    result = VarianceCollapseDetector("medium").check(store)
    assert result.status == Status.OK


def test_variance_insufficient_data():
    store = MetricStore("t", window_size=50)
    feed(store, [0.5] * 10)
    assert VarianceCollapseDetector().check(store).status == Status.INSUFFICIENT_DATA


# -- length --------------------------------------------------------------------

def test_length_drift_fires():
    store = MetricStore("t", window_size=50)
    lengths = [100 + (i % 5) for i in range(50)] + [400 + (i % 5) for i in range(50)]
    feed(store, [1.0] * 100, lengths=lengths)
    result = LengthDriftDetector("medium").check(store)
    assert result.status == Status.WARNING
    assert "longer" in result.message


def test_length_stable_is_ok():
    store = MetricStore("t", window_size=50)
    lengths = [100 + (i % 5) for i in range(200)]
    feed(store, [1.0] * 200, lengths=lengths)
    assert LengthDriftDetector("medium").check(store).status == Status.OK


# -- slope (CUSUM) -------------------------------------------------------------

def test_slope_change_fires_on_sustained_shift():
    store = MetricStore("t", window_size=100)
    detector = RewardSlopeChangeDetector("medium")
    fired = False
    rewards = [0.2] * 30 + [0.9] * 60
    for i, r in enumerate(rewards):
        store.append(make_record(i, r))
        if detector.check(store, store.records[-1]).status == Status.WARNING:
            fired = True
    assert fired


def test_slope_stable_does_not_fire():
    store = MetricStore("t", window_size=100)
    detector = RewardSlopeChangeDetector("medium")
    fired = False
    for i in range(120):
        store.append(make_record(i, 0.5))
        if detector.check(store, store.records[-1]).status == Status.WARNING:
            fired = True
    assert not fired


# -- component dominance -------------------------------------------------------

def test_component_dominance_fires():
    store = MetricStore("t", window_size=50)
    comps = [{"correctness": 0.0, "format": 1.0, "total": 1.0} for _ in range(60)]
    feed(store, [1.0] * 60, components_seq=comps)
    result = ComponentDominanceDetector("medium").check(store)
    assert result.status == Status.ALERT
    assert "format" in result.message


def test_component_balanced_is_ok():
    store = MetricStore("t", window_size=50)
    comps = [{"correctness": 0.5, "format": 0.5, "total": 1.0} for _ in range(60)]
    feed(store, [1.0] * 60, components_seq=comps)
    assert ComponentDominanceDetector("medium").check(store).status == Status.OK


def test_component_single_is_not_applicable():
    store = MetricStore("t", window_size=50)
    comps = [{"correctness": 1.0, "total": 1.0} for _ in range(60)]
    feed(store, [1.0] * 60, components_seq=comps)
    assert ComponentDominanceDetector().check(store).status == Status.NOT_APPLICABLE


# -- ceiling -------------------------------------------------------------------

def test_ceiling_saturation_fires():
    store = MetricStore("t", window_size=50)
    feed(store, [1.0] * 60)
    result = CeilingRateDetector("medium").check(store)
    assert result.status == Status.ALERT


def test_ceiling_spread_is_ok():
    store = MetricStore("t", window_size=50)
    feed(store, [(i % 10) / 10.0 for i in range(100)])
    assert CeilingRateDetector("medium").check(store).status == Status.OK


# -- engine --------------------------------------------------------------------

def test_engine_records_alerts_and_dedupes():
    store = MetricStore("t", window_size=50)
    engine = DetectionEngine(store, sensitivity="medium")
    for i in range(120):
        store.append(make_record(i, 1.0, {"correctness": 0.0, "format": 1.0, "total": 1.0}))
        engine.process(store.records[-1])
    # Multiple detectors should fire (ceiling, component, variance) but each
    # ongoing condition is recorded once, not once per step.
    assert len(store.alerts) > 0
    assert len(store.alerts) < 20
    assert engine.overall in (Status.WARNING, Status.ALERT)
    assert store.engine is engine


def test_engine_fires_callback():
    store = MetricStore("t", window_size=50)
    seen = []
    engine = DetectionEngine(store, callbacks=[seen.append])
    for i in range(120):
        store.append(make_record(i, 1.0))
        engine.process(store.records[-1])
    assert len(seen) >= 1


def test_watch_runs_detection_end_to_end():
    @rewardspy.watch(name="hacky_reward")
    def reward(response, gt):
        return {"correctness": 0.0, "format": 1.0, "total": 1.0}

    for _ in range(120):
        reward("same short response", "gt")

    assert reward.engine is not None
    assert len(reward.store.alerts) > 0


def test_detect_false_skips_engine():
    @rewardspy.watch(name="silent_reward", detect=False)
    def reward(response, gt):
        return 1.0

    for _ in range(60):
        reward("r", "g")
    assert reward.engine is None
    assert reward.store.alerts == []
