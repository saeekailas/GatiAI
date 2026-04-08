from baseline import RuleBasedAgent
from agent_evaluator import AgentEvaluator


def test_agent_evaluator_runs_all_tasks():
    agent = RuleBasedAgent()
    evaluator = AgentEvaluator(tasks=["task1", "task2", "task3"], seeds=[42])
    report = evaluator.evaluate(agent, verbose=False)

    assert report["aggregate"]["runs"] == 3
    assert report["aggregate"]["average_reward"] >= 0.0
    assert report["aggregate"]["success_rate"] >= 0.0
    assert "tasks" in report
    assert len(report["tasks"]) == 3
    for task_summary in report["tasks"]:
        assert task_summary["runs"] == 1
        assert task_summary["average_reward"] >= 0.0
        assert 0.0 <= task_summary["success_rate"] <= 1.0
