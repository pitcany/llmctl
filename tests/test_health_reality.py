from llmctl.schemas import HealthState


def test_not_used_state():
    assert HealthState.NOT_USED.value == "not_used"
    assert HealthState("not_used") is HealthState.NOT_USED
