from environment import SupplyChainEnv
from models import ActionType, SCAction


def test_environment_reset_and_state():
    env = SupplyChainEnv(task_id="task1", seed=42)
    obs = env.reset(seed=42)
    assert obs.task_id == "task1"
    assert obs.turn == 0
    assert not obs.done
    state = env.state()
    assert state.task_id == "task1"
    assert state.turn == 0
    assert state.current_score == 0.0


def test_environment_step_task1():
    env = SupplyChainEnv(task_id="task1", seed=42)
    obs = env.reset(seed=42)
    action = SCAction(
        action_type=ActionType.SELECT_SUPPLIER,
        target_id=obs.available_suppliers[0].supplier_id,
        parameters={"quantity": 500},
        explanation="Selecting a supplier to resolve disruption.",
    )
    obs, reward, done, info = env.step(action)
    assert isinstance(obs.turn, int)
    assert obs.last_action_result is not None
    assert done in (True, False)
    assert reward == 0.0 or reward >= 0.0
    assert "total_cost" in info


def test_environment_step_task3_partial_info():
    env = SupplyChainEnv(task_id="task3", seed=7)
    obs = env.reset(seed=7)
    assert obs.partial_info is True
    action = SCAction(
        action_type=ActionType.REQUEST_INFO,
        parameters={"query_key": "stock_query_steel"},
        explanation="Request stock details before making a decision.",
    )
    obs, reward, done, info = env.step(action)
    assert obs.info_available
    assert obs.last_action_result is not None
    assert reward == 0.0
    assert done is False or done is True
