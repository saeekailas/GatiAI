"""
GatiAI — Inference Script for OpenEnv Hackathon
Runs a simple rule-based agent on all 3 tasks to produce reproducible baseline scores.
Run: python inference.py
"""
from __future__ import annotations
import json
import os
from models import SCAction, ActionType
from environment import SupplyChainEnv

# Required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")  # This is the OpenAI API key

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client with required vars
from openai import OpenAI
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ─── Simple rule-based baseline agent ────────────────────────────────────────

class RuleBasedAgent:
    """
    Greedy rule-based agent.
    Task1: Picks cheapest available supplier with lead_time < 5 days.
    Task2: Expedites critical, reroutes medium, delays low urgency.
    Task3: Requests stock info then picks cheapest available supplier.
    """

    def act(self, obs, task_id: str) -> SCAction:
        if task_id == "task1":
            return self._act_task1(obs)
        elif task_id == "task2":
            return self._act_task2(obs)
        elif task_id == "task3":
            return self._act_task3(obs)
        raise ValueError(f"Unknown task: {task_id}")

    def _act_task1(self, obs) -> SCAction:
        available = [s for s in obs.available_suppliers if s.available]
        if not available:
            return SCAction(
                action_type = ActionType.ESCALATE_TO_HUMAN,
                explanation = "No suppliers available, escalating.",
            )
        # Pick cheapest with acceptable lead time
        candidates = [s for s in available if s.lead_time_days <= 5]
        if not candidates:
            candidates = available
        best = min(candidates, key=lambda s: s.cost_per_unit)
        return SCAction(
            action_type = ActionType.SELECT_SUPPLIER,
            target_id   = best.supplier_id,
            parameters  = {"quantity": 500},
            explanation = (
                f"Selected {best.name} because it is the cheapest available "
                f"supplier at ₹{best.cost_per_unit}/unit with {best.lead_time_days}-day "
                f"lead time and reliability {best.reliability_score}."
            ),
        )

    def _act_task2(self, obs) -> SCAction:
        for shp in obs.affected_shipments:
            if shp.urgency.value == "critical":
                return SCAction(
                    action_type = ActionType.EXPEDITE_SHIPMENT,
                    target_id   = shp.shipment_id,
                    explanation = (
                        f"Expediting {shp.cargo} shipment due to CRITICAL urgency "
                        f"and {shp.deadline_days}-day hard deadline."
                    ),
                )
            elif shp.urgency.value == "medium":
                return SCAction(
                    action_type = ActionType.REROUTE_SHIPMENT,
                    target_id   = shp.shipment_id,
                    explanation = (
                        f"Rerouting {shp.cargo} to avoid port congestion "
                        f"at lower cost than expediting."
                    ),
                )
            elif shp.urgency.value == "low":
                return SCAction(
                    action_type = ActionType.DELAY_ORDER,
                    target_id   = shp.shipment_id,
                    parameters  = {"delay_days": 7},
                    explanation = "Delaying low-urgency shipment to preserve budget.",
                )
        return SCAction(
            action_type = ActionType.ESCALATE_TO_HUMAN,
            explanation = "No clear action, escalating.",
        )

    def _act_task3(self, obs) -> SCAction:
        # First: request info if we haven't yet
        if obs.partial_info and not obs.info_available:
            return SCAction(
                action_type = ActionType.REQUEST_INFO,
                parameters  = {"query_key": "stock_query_steel"},
                explanation = (
                    "Requesting actual stock levels before committing "
                    "because inventory data is unknown under partial info conditions."
                ),
            )
        if obs.partial_info and len(obs.info_available) == 1:
            return SCAction(
                action_type = ActionType.REQUEST_INFO,
                parameters  = {"query_key": "vendor_status_query"},
                explanation = (
                    "Requesting vendor status to confirm availability "
                    "before selecting supplier under uncertainty."
                ),
            )
        # Then act
        available = [s for s in obs.available_suppliers if s.available]
        if available:
            best = min(available, key=lambda s: s.cost_per_unit)
            return SCAction(
                action_type = ActionType.SELECT_SUPPLIER,
                target_id   = best.supplier_id,
                parameters  = {"quantity": 400},
                explanation = (
                    f"After gathering inventory and vendor info, selected {best.name} "
                    f"as cheapest reliable available supplier to prevent production stoppage. "
                    f"Risk: moderate — verified via prior info requests."
                ),
            )
        return SCAction(
            action_type = ActionType.ESCALATE_TO_HUMAN,
            explanation = "No available suppliers found after info gathering — escalating to human.",
        )


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_inference(task_id: str, seed: int = 42, verbose: bool = True) -> dict:
    print(f"[START] Task: {task_id}, Seed: {seed}")
    env   = SupplyChainEnv(task_id=task_id, seed=seed)
    agent = RuleBasedAgent()
    obs   = env.reset(seed=seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK: {task_id.upper()} | Seed: {seed}")
        print(f"{'='*60}")
        print(f"Disruption: {obs.disruption_type.value}")
        print(f"Description: {obs.disruption_description[:120]}...")

    episode_result = None
    turn           = 0

    while not obs.done:
        action = agent.act(obs, task_id)
        print(f"[STEP] Turn: {turn+1}, Action: {action.action_type.value}, Target: {action.target_id}")
        obs, reward, done, info = env.step(action)
        turn += 1

        if verbose:
            print(f"\n  Turn {turn}: {action.action_type.value}"
                  f" → target={action.target_id}")
            print(f"  Result: {obs.last_action_result}")

        if done and "episode_result" in info:
            episode_result = info["episode_result"]
            print(f"[END] Final Reward: {episode_result.final_reward:.3f}, Success: {episode_result.success}, Cost: ₹{episode_result.total_cost:,.0f}, Turns: {episode_result.turns_taken}")

    if episode_result and verbose:
        print(f"\n{'─'*60}")
        print(f"FINAL REWARD:  {episode_result.final_reward:.3f}")
        print(f"SUCCESS:       {episode_result.success}")
        print(f"TOTAL COST:    ₹{episode_result.total_cost:,.0f}")
        print(f"TURNS TAKEN:   {episode_result.turns_taken}")
        print(f"\nSCORE BREAKDOWN:")
        for dim, score in episode_result.score_breakdown.items():
            bar = "█" * int(score * 20)
            print(f"  {dim:<30} {score:.3f}  {bar}")
        print(f"\nGRADER FEEDBACK:\n  {episode_result.grader_feedback}")

    return episode_result.model_dump() if episode_result else {}


def run_all_inference():
    """Run all 3 tasks and print a summary table."""
    print("\n" + "="*60)
    print(" SUPPLY CHAIN DISRUPTION GYM — INFERENCE SCORES")
    print("="*60)

    results = {}
    for task_id in ["task1", "task2", "task3"]:
        r = run_inference(task_id=task_id, seed=42, verbose=True)
        results[task_id] = r

    print("\n\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"{'Task':<10} {'Reward':<10} {'Success':<10} {'Cost (₹)':<15} {'Turns'}")
    print("─"*55)
    for task_id, r in results.items():
        if r:
            print(f"{task_id:<10} {r['final_reward']:<10.3f} "
                  f"{str(r['success']):<10} {r['total_cost']:<15,.0f} "
                  f"{r['turns_taken']}")
    print("="*60)

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to inference_results.json")


if __name__ == "__main__":
    run_all_inference()
