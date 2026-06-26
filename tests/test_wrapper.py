import warnings

import pytest

import rewardspy
from rewardspy.wrapper import registered_stores


def test_one_line_wrap_is_transparent():
    def reward(response, gt):
        return 0.7

    watched = rewardspy.watch(reward)
    assert watched("hello", "world") == 0.7
    assert watched.store.count == 1
    assert watched.store.rolling_mean == pytest.approx(0.7)


def test_bare_decorator():
    @rewardspy.watch
    def reward(response, gt):
        return float(len(response))

    assert reward("abcd", "x") == 4.0
    assert reward.store.count == 1


def test_preserves_metadata():
    def my_reward(response, gt):
        """scores things"""
        return 1.0

    watched = rewardspy.watch(my_reward)
    assert watched.__name__ == "my_reward"
    assert watched.__doc__ == "scores things"


def test_dict_return_tracks_components_and_total():
    @rewardspy.watch(name="math", components=["correctness", "format"])
    def reward(response, answer):
        return {"correctness": 0.8, "format": 0.1, "total": 0.9}

    assert reward("r", "a") == {"correctness": 0.8, "format": 0.1, "total": 0.9}
    store = reward.store
    assert store.rolling_mean == pytest.approx(0.9)  # reads "total"
    assert store.component_stats["correctness"].rolling_mean == pytest.approx(0.8)
    assert store.component_stats["format"].rolling_mean == pytest.approx(0.1)


def test_dict_without_total_sums_components():
    @rewardspy.watch(components=["a", "b"])
    def reward(response, gt):
        return {"a": 0.3, "b": 0.4}

    reward("r", "g")
    assert reward.store.rolling_mean == pytest.approx(0.7)


def test_return_value_is_unchanged_object():
    payload = {"correctness": 1.0, "total": 1.0}

    @rewardspy.watch
    def reward(response, gt):
        return payload

    assert reward("r", "g") is payload


def test_step_counter_advances():
    @rewardspy.watch
    def reward(response, gt):
        return 0.5

    for _ in range(5):
        reward("r", "g")
    assert reward.store.step_counter == 5


def test_response_length_uses_first_argument():
    @rewardspy.watch
    def reward(response, gt):
        return 1.0

    reward("a response of some length", "ground truth")
    record = reward.store.recent(1)[0]
    assert record.output_length == len("a response of some length")


def test_exceptions_propagate_and_are_not_recorded():
    @rewardspy.watch
    def reward(response, gt):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        reward("r", "g")
    assert reward.store.count == 0


def test_unparseable_return_warns_and_passes_through():
    @rewardspy.watch
    def reward(response, gt):
        return object()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reward("r", "g")
    assert isinstance(result, object)
    assert reward.store.count == 0
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_session_groups_stores():
    spy = rewardspy.Session(name="run1", window_size=50)

    @spy.watch(components=["correctness"])
    def reward(response, gt):
        return {"correctness": 1.0, "total": 1.0}

    reward("r", "g")
    assert len(spy.stores) == 1
    store = next(iter(spy.stores.values()))
    assert store.window_size == 50
    assert store.count == 1


def test_registry_discovers_stores():
    name = "discoverable_reward_fn_xyz"

    @rewardspy.watch(name=name)
    def reward(response, gt):
        return 1.0

    reward("r", "g")
    assert name in registered_stores()
    assert registered_stores()[name].count == 1
