"""
GatiAI Agent Evaluator

Standard evaluation harness for agents using the GatiAI OpenEnv gym.
It runs agents across tasks, seeds, and metrics, and produces rich per-task
reports that expose the environment's multi-dimensional grading.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from baseline import RuleBasedAgent
from environment import SupplyChainEnv
from models import EpisodeResult, SCAction


@dataclass
class EvaluationRecord:
    task_id: str
    seed: int
    episode_result: EpisodeResult
    actions: List[SCAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "episode_result": self.episode_result.model_dump(),
            "actions": [a.model_dump() for a in self.actions],
        }


class AgentEvaluator:
    """Runs a policy against the GatiAI environment and summarizes results."""

    DEFAULT_TASKS = ["task1", "task2", "task3"]
    DEFAULT_SEEDS = [42, 99, 123]

    def __init__(self, tasks: Optional[Sequence[str]] = None, seeds: Optional[Sequence[int]] = None):
        self.tasks = list(tasks or self.DEFAULT_TASKS)
        self.seeds = list(seeds or self.DEFAULT_SEEDS)

    def evaluate(self, agent: object, verbose: bool = False) -> Dict[str, Any]:
        results: Dict[str, List[Dict[str, Any]]] = {}
        for task_id in self.tasks:
            task_results = []
            for seed in self.seeds:
                record = self._evaluate_single_run(agent, task_id, seed, verbose)
                task_results.append(record.to_dict())
            results[task_id] = task_results
        return self._summarize(results)

    def _evaluate_single_run(self, agent: object, task_id: str, seed: int, verbose: bool) -> EvaluationRecord:
        env = SupplyChainEnv(task_id=task_id, seed=seed)
        obs = env.reset(seed=seed)
        actions: List[SCAction] = []

        if verbose:
            print(f"Evaluating {task_id} (seed={seed})")

        while not obs.done:
            action = agent.act(obs, task_id)
            if not isinstance(action, SCAction):
                raise ValueError("Agent.act() must return an SCAction instance")

            obs, reward, done, info = env.step(action)
            actions.append(action)
            if verbose:
                print(f"  turn={obs.turn:2d} action={action.action_type.value} target={action.target_id} done={done}")

        episode_result = info.get("episode_result")
        if episode_result is None:
            raise RuntimeError("Episode completed without an episode_result")

        return EvaluationRecord(
            task_id=task_id,
            seed=seed,
            episode_result=episode_result,
            actions=actions,
        )

    def _summarize(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        summary = {
            "tasks": [],
            "aggregate": {
                "runs": 0,
                "average_reward": 0.0,
                "success_rate": 0.0,
            },
        }
        total_reward = 0.0
        total_success = 0
        total_runs = 0

        for task_id, runs in results.items():
            task_reward = sum(run["episode_result"]["final_reward"] for run in runs)
            task_success = sum(1 for run in runs if run["episode_result"]["success"])
            count = len(runs)
            summary["tasks"].append({
                "task_id": task_id,
                "runs": count,
                "average_reward": round(task_reward / max(count, 1), 3),
                "success_rate": round(task_success / max(count, 1), 3),
            })
            total_reward += task_reward
            total_success += task_success
            total_runs += count

        summary["aggregate"]["runs"] = total_runs
        summary["aggregate"]["average_reward"] = round(total_reward / max(total_runs, 1), 3)
        summary["aggregate"]["success_rate"] = round(total_success / max(total_runs, 1), 3)
        summary["results"] = results
        return summary


def run_default_evaluation() -> None:
    agent = RuleBasedAgent()
    evaluator = AgentEvaluator()
    report = evaluator.evaluate(agent, verbose=True)
    print("\n===== EVALUATION SUMMARY =====")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    run_default_evaluation()
