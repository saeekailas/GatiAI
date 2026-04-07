"""
Supply Chain Disruption Gym — Core Environment
Implements the OpenEnv spec: reset() / step() / state()
"""
from __future__ import annotations
import copy
from typing import Optional

from models import (
    SCAction, SCObservation, SCState, EpisodeResult,
    ActionType, DisruptionType
)
from scenarios import get_scenario
from graders import get_grader


class SupplyChainEnv:
    """
    Full OpenEnv-spec environment for supply chain disruption training.

    Usage:
        env = SupplyChainEnv(task_id="task1")
        obs = env.reset()
        while not obs.done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        result = info["episode_result"]
    """

    VALID_TASKS = ("task1", "task2", "task3")

    def __init__(self, task_id: str = "task1", seed: Optional[int] = None):
        if task_id not in self.VALID_TASKS:
            raise ValueError(f"task_id must be one of {self.VALID_TASKS}")
        self.task_id     = task_id
        self.seed        = seed
        self._scenario   = None
        self._grader     = get_grader(task_id)
        self._actions    = []
        self._turn       = 0
        self._done       = False
        self._total_cost = 0.0
        self._production_stopped = True
        self._info_revealed: dict = {}

    # ─── reset() ──────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> SCObservation:
        """Reset to a fresh episode. Returns initial observation."""
        _seed           = seed or self.seed
        self._scenario  = get_scenario(self.task_id, seed=_seed)
        self._actions   = []
        self._turn      = 0
        self._done      = False
        self._total_cost = 0.0
        self._production_stopped = True
        self._info_revealed = {}

        return self._make_observation(last_result=None)

    # ─── step() ───────────────────────────────────────────────────────────────

    def step(self, action: SCAction) -> tuple[SCObservation, float, bool, dict]:
        """
        Execute one agent action.
        Returns: (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._scenario is None:
            raise RuntimeError("Call reset() before step().")

        self._turn += 1
        self._actions.append(action)

        result_message = self._execute_action(action)

        # Check terminal conditions
        max_turns = self._scenario["max_turns"]
        terminal  = self._turn >= max_turns or self._done

        if terminal:
            self._done = True
            episode_result = self._grade_episode()
            reward = episode_result.final_reward
            info   = {
                "episode_result": episode_result,
                "turn": self._turn,
                "total_cost": self._total_cost,
            }
        else:
            reward = 0.0   # reward only given at end (GRPO style)
            info   = {"turn": self._turn, "total_cost": self._total_cost}

        obs = self._make_observation(last_result=result_message)
        obs.done = self._done

        return obs, reward, self._done, info

    # ─── state() ──────────────────────────────────────────────────────────────

    def state(self) -> SCState:
        """Return current environment state snapshot."""
        if self._scenario is None:
            raise RuntimeError("Call reset() before state().")
        return SCState(
            task_id             = self.task_id,
            turn                = self._turn,
            max_turns           = self._scenario["max_turns"],
            disruption_type     = self._scenario["disruption_type"],
            total_cost_incurred = self._total_cost,
            production_stopped  = self._production_stopped,
            orders_delayed      = sum(
                1 for a in self._actions
                if a.action_type == ActionType.DELAY_ORDER),
            escalated           = any(
                a.action_type == ActionType.ESCALATE_TO_HUMAN
                for a in self._actions),
            actions_taken       = [
                {"type": a.action_type, "target": a.target_id,
                 "params": a.parameters, "explanation": a.explanation}
                for a in self._actions],
            current_score       = self._compute_partial_score(),
            done                = self._done,
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _execute_action(self, action: SCAction) -> str:
        """Mutate environment state based on action. Returns result message."""
        sc = self._scenario

        if action.action_type == ActionType.SELECT_SUPPLIER:
            supplier = next(
                (s for s in sc["available_suppliers"]
                 if s.supplier_id == action.target_id), None)
            if supplier and supplier.available:
                cost = supplier.cost_per_unit * action.parameters.get("quantity", 500)
                self._total_cost += cost
                self._production_stopped = False
                return (f"Supplier {supplier.name} selected. "
                        f"Cost: ₹{cost:,.2f}. Lead time: {supplier.lead_time_days} days.")
            return f"Supplier {action.target_id} not available or not found."

        elif action.action_type == ActionType.REROUTE_SHIPMENT:
            shp = next(
                (s for s in sc.get("affected_shipments", [])
                 if s.shipment_id == action.target_id), None)
            if shp:
                self._total_cost += shp.cost_to_reroute
                return (f"Shipment {shp.shipment_id} ({shp.cargo}) rerouted. "
                        f"Cost: ₹{shp.cost_to_reroute:,.2f}.")
            return f"Shipment {action.target_id} not found."

        elif action.action_type == ActionType.EXPEDITE_SHIPMENT:
            shp = next(
                (s for s in sc.get("affected_shipments", [])
                 if s.shipment_id == action.target_id), None)
            if shp:
                self._total_cost += shp.cost_to_expedite
                return (f"Shipment {shp.shipment_id} ({shp.cargo}) expedited. "
                        f"Cost: ₹{shp.cost_to_expedite:,.2f}. "
                        f"Will meet {shp.deadline_days}-day deadline.")
            return f"Shipment {action.target_id} not found."

        elif action.action_type == ActionType.DELAY_ORDER:
            shp = next(
                (s for s in sc.get("affected_shipments", [])
                 if s.shipment_id == action.target_id), None)
            if shp:
                delay_days = action.parameters.get("delay_days", 7)
                penalty    = shp.delay_penalty * delay_days
                self._total_cost += penalty
                return (f"Order {shp.shipment_id} delayed by {delay_days} days. "
                        f"Penalty: ₹{penalty:,.2f}.")
            return f"Shipment {action.target_id} not found."

        elif action.action_type == ActionType.REQUEST_INFO:
            query = action.parameters.get("query_key", "")
            hidden = sc.get("hidden_info", {})
            if query in hidden:
                self._info_revealed[query] = hidden[query]
                return f"Info retrieved for '{query}': {hidden[query]}"
            # Return available query keys as hint
            available_keys = list(hidden.keys())
            return (f"Query '{query}' not found. "
                    f"Available queries: {available_keys}")

        elif action.action_type == ActionType.ESCALATE_TO_HUMAN:
            return ("Escalated to human supervisor. "
                    "Awaiting human decision — production on hold.")

        elif action.action_type == ActionType.SPLIT_ORDER:
            supplier_ids = action.parameters.get("supplier_ids", [])
            valid = [
                s for s in sc["available_suppliers"]
                if s.supplier_id in supplier_ids and s.available]
            if valid:
                cost = sum(s.cost_per_unit * 250 for s in valid)
                self._total_cost += cost
                self._production_stopped = False
                return (f"Order split across {len(valid)} suppliers. "
                        f"Total cost: ₹{cost:,.2f}.")
            return "No valid suppliers in split order."

        return f"Unknown action: {action.action_type}"

    def _make_observation(self, last_result: Optional[str]) -> SCObservation:
        sc = self._scenario
        return SCObservation(
            turn                    = self._turn,
            task_id                 = self.task_id,
            disruption_type         = sc["disruption_type"],
            disruption_description  = sc["disruption_description"],
            available_suppliers     = sc["available_suppliers"],
            affected_shipments      = sc.get("affected_shipments", []),
            warehouse_stocks        = sc["warehouse_stocks"],
            budget_remaining        = sc["budget_remaining"] - self._total_cost,
            time_remaining_days     = max(0, sc["time_remaining_days"] - self._turn),
            partial_info            = sc.get("partial_info", False),
            info_available          = self._info_revealed,
            last_action_result      = last_result,
            done                    = self._done,
            message                 = (
                f"Turn {self._turn}/{sc['max_turns']}. "
                f"Budget remaining: ₹{sc['budget_remaining'] - self._total_cost:,.0f}."
            ),
        )

    def _grade_episode(self) -> EpisodeResult:
        final_state = {
            "turns_taken":        self._turn,
            "total_cost":         self._total_cost,
            "production_stopped": self._production_stopped,
        }
        return self._grader.grade(self._actions, self._scenario, final_state)

    def _compute_partial_score(self) -> float:
        """Rough mid-episode score for monitoring."""
        if not self._actions:
            return 0.0
        positive = sum(1 for a in self._actions
                       if a.action_type in (
                           ActionType.SELECT_SUPPLIER,
                           ActionType.EXPEDITE_SHIPMENT,
                           ActionType.REROUTE_SHIPMENT,
                           ActionType.SPLIT_ORDER,
                       ))
        return min(positive * 0.15, 0.75)