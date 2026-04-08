"""
GatiAI Demo Script
Interactive demonstration of the Supply Chain Disruption Gym.
Shows how agents interact with the environment and get graded.

Run: python demo.py
"""
from __future__ import annotations
import json
from models import SCAction, ActionType
from environment import SupplyChainEnv


class SmartAgent:
    """
    Intelligent agent that reasons about supply chain decisions.
    Better than baseline: uses cost/lead-time analysis and risk assessment.
    """

    def act(self, obs, task_id: str) -> SCAction:
        """Choose action based on observation."""
        if task_id == "task1":
            return self._smart_task1(obs)
        elif task_id == "task2":
            return self._smart_task2(obs)
        elif task_id == "task3":
            return self._smart_task3(obs)
        raise ValueError(f"Unknown task: {task_id}")

    def _smart_task1(self, obs) -> SCAction:
        """Task 1: Smart supplier selection balancing cost, lead time, reliability."""
        available = [s for s in obs.available_suppliers if s.available]
        if not available:
            return SCAction(
                action_type=ActionType.ESCALATE_TO_HUMAN,
                explanation="No suppliers available. Critical situation.",
            )

        # Score suppliers: cost efficiency + lead time speed + reliability
        def score_supplier(s):
            cost_score = 1.0 / max(s.cost_per_unit, 1)
            speed_score = 1.0 / max(s.lead_time_days, 1)
            reliability_score = s.reliability_score
            # Weighted: 40% cost, 30% speed, 30% reliability
            return (cost_score * 0.4 + speed_score * 0.3 + reliability_score * 0.3)

        best = max(available, key=score_supplier)
        return SCAction(
            action_type=ActionType.SELECT_SUPPLIER,
            target_id=best.supplier_id,
            parameters={"quantity": 500},
            explanation=(
                f"Selected {best.name} (score: {score_supplier(best):.3f}) "
                f"balancing cost (₹{best.cost_per_unit}), "
                f"lead time ({best.lead_time_days}d), and "
                f"reliability ({best.reliability_score}). "
                f"This optimizes total supply chain efficiency."
            ),
        )

    def _smart_task2(self, obs) -> SCAction:
        """Task 2: Prioritize critical shipments under budget constraint."""
        critical_shipments = [s for s in obs.affected_shipments 
                              if s.urgency.value == "critical"]
        high_shipments = [s for s in obs.affected_shipments 
                          if s.urgency.value == "high"]
        
        if critical_shipments:
            target = critical_shipments[0]
            cost_expedite = target.cost_to_expedite
            cost_reroute = target.cost_to_reroute
            
            if cost_expedite <= obs.budget_remaining:
                return SCAction(
                    action_type=ActionType.EXPEDITE_SHIPMENT,
                    target_id=target.shipment_id,
                    explanation=(
                        f"CRITICAL: Expediting {target.cargo} to meet "
                        f"{target.deadline_days}-day deadline. "
                        f"Cost ₹{cost_expedite:.0f} is within budget. "
                        f"Failure cost would exceed this by 100x."
                    ),
                )
            elif cost_reroute <= obs.budget_remaining:
                return SCAction(
                    action_type=ActionType.REROUTE_SHIPMENT,
                    target_id=target.shipment_id,
                    explanation=(
                        f"Rerouting {target.cargo} at cost ₹{cost_reroute:.0f}. "
                        f"Expedite too expensive; reroute is next best option."
                    ),
                )
        
        if high_shipments:
            target = high_shipments[0]
            return SCAction(
                action_type=ActionType.REROUTE_SHIPMENT,
                target_id=target.shipment_id,
                explanation=(
                    f"Rerouting {target.cargo} to handle high-urgency shipment "
                    f"within budget constraints."
                ),
            )
        
        return SCAction(
            action_type=ActionType.ESCALATE_TO_HUMAN,
            explanation="Unable to handle all shipments with current budget.",
        )

    def _smart_task3(self, obs) -> SCAction:
        """Task 3: Gather info under uncertainty, then decide wisely."""
        if obs.partial_info and len(obs.info_available) < 1:
            return SCAction(
                action_type=ActionType.REQUEST_INFO,
                parameters={"query_key": "stock_query_steel"},
                explanation=(
                    "Under partial observability, requesting critical inventory data "
                    "before committing resources. Production stoppage penalty is ₹200k/day."
                ),
            )
        
        if obs.partial_info and len(obs.info_available) < 2:
            return SCAction(
                action_type=ActionType.REQUEST_INFO,
                parameters={"query_key": "vendor_status_query"},
                explanation=(
                    "Requesting vendor status to confirm reliability before selection. "
                    "This reduces risk in a multi-failure scenario."
                ),
            )
        
        available = [s for s in obs.available_suppliers if s.available]
        if available:
            # Prefer high reliability over low cost in crisis
            best = max(available, key=lambda s: s.reliability_score * 0.6 + 
                       (1.0 / max(s.cost_per_unit, 1)) * 0.4)
            return SCAction(
                action_type=ActionType.SELECT_SUPPLIER,
                target_id=best.supplier_id,
                parameters={"quantity": 400},
                explanation=(
                    f"After information gathering, selected {best.name} "
                    f"with reliability {best.reliability_score}. "
                    f"Risk-aware decision under multi-failure crisis conditions."
                ),
            )
        
        return SCAction(
            action_type=ActionType.ESCALATE_TO_HUMAN,
            explanation="Crisis escalates — no viable suppliers. Human intervention needed.",
        )


def demo_single_task(task_id: str, agent_name: str = "SmartAgent", seed: int = 42):
    """Run a single task and display results."""
    print(f"\n{'='*70}")
    print(f"  TASK: {task_id.upper()} | Agent: {agent_name} | Seed: {seed}")
    print(f"{'='*70}\n")
    
    env = SupplyChainEnv(task_id=task_id, seed=seed)
    agent = SmartAgent()
    obs = env.reset(seed=seed)
    
    print(f"📌 DISRUPTION TYPE: {obs.disruption_type.value.upper()}")
    print(f"📌 DESCRIPTION:")
    print(f"   {obs.disruption_description}\n")
    
    print(f"📊 INITIAL STATE:")
    print(f"   Budget: ₹{obs.budget_remaining:,.0f}")
    print(f"   Time: {obs.time_remaining_days} days")
    print(f"   Turn: {obs.turn}/{env._scenario['max_turns']}")
    print(f"   Suppliers available: {sum(1 for s in obs.available_suppliers if s.available)}")
    print(f"   Affected shipments: {len(obs.affected_shipments)}\n")
    
    episode_result = None
    turn = 0
    
    print(f"{'─'*70}")
    print("  AGENT ACTIONS")
    print(f"{'─'*70}\n")
    
    while not obs.done and turn < env._scenario['max_turns']:
        action = agent.act(obs, task_id)
        obs, reward, done, info = env.step(action)
        turn += 1
        
        print(f"Turn {turn}:")
        print(f"  Action: {action.action_type.value}")
        if action.target_id:
            print(f"  Target: {action.target_id}")
        print(f"  Explanation: {action.explanation}")
        print(f"  Result: {obs.last_action_result}")
        print(f"  Budget left: ₹{obs.budget_remaining:,.0f}\n")
        
        if done and "episode_result" in info:
            episode_result = info["episode_result"]
    
    if episode_result:
        print(f"{'─'*70}")
        print("  FINAL RESULTS")
        print(f"{'─'*70}\n")
        print(f"✓ FINAL REWARD: {episode_result.final_reward:.3f}/1.0")
        print(f"✓ SUCCESS: {episode_result.success}")
        print(f"✓ TOTAL COST: ₹{episode_result.total_cost:,.0f}")
        print(f"✓ TURNS TAKEN: {episode_result.turns_taken}")
        print(f"\n📈 SCORE BREAKDOWN:")
        for dim, score in episode_result.score_breakdown.items():
            bar = "█" * int(score * 20)
            print(f"   {dim:<30} {score:.3f}  {bar}")
        print(f"\n💬 GRADER FEEDBACK:")
        print(f"   {episode_result.grader_feedback}\n")
    
    return episode_result


def demo_all_tasks():
    """Run all tasks and show comparison."""
    print("\n" + "="*70)
    print("  GATIAI SUPPLY CHAIN DISRUPTION GYM — DEMO")
    print("="*70)
    print("\nThis demo shows how an intelligent agent performs across different")
    print("supply chain crisis scenarios. Each task tests different capabilities:\n")
    print("  • Task 1 (Easy):        Single supplier failure — decision analysis")
    print("  • Task 2 (Medium):      Port congestion — multi-stakeholder prioritization")
    print("  • Task 3 (Hard):        Multi-vendor crisis — reasoning under uncertainty\n")
    
    results = {}
    for task_id in ["task1", "task2", "task3"]:
        result = demo_single_task(task_id, agent_name="SmartAgent", seed=42)
        results[task_id] = result
    
    print("\n" + "="*70)
    print("  SUMMARY ACROSS ALL TASKS")
    print("="*70 + "\n")
    print(f"{'Task':<10} {'Reward':<12} {'Success':<12} {'Cost (₹)':<15} {'Turns':<8}")
    print("─" * 60)
    for task_id in ["task1", "task2", "task3"]:
        r = results.get(task_id, {})
        if r:
            reward = r.get("final_reward", 0.0)
            success = r.get("success", False)
            cost = r.get("total_cost", 0.0)
            turns = r.get("turns_taken", 0)
            print(f"{task_id:<10} {reward:<12.3f} {str(success):<12} {cost:<15,.0f} {turns:<8}")
    print("=" * 60)
    
    print("\n💡 KEY INSIGHTS:")
    print("   • Reward function grades multiple dimensions (not just correctness)")
    print("   • Cost efficiency matters as much as decision accuracy")
    print("   • Explanation quality contributes to final score")
    print("   • Agents must balance speed (fewer turns) with quality decisions\n")
    
    print("🚀 Want to test your own agent?")
    print("   1. Implement act(obs, task_id) -> SCAction in your agent class")
    print("   2. Run: env = SupplyChainEnv(task_id='task1')")
    print("   3. obs = env.reset()")
    print("   4. Loop: obs, reward, done, info = env.step(action)")
    print("   5. Check episode_result in info when done=True\n")


def demo_api_style():
    """Show REST API interaction style."""
    print("\n" + "="*70)
    print("  API-STYLE DEMO (How to interact via HTTP)")
    print("="*70 + "\n")
    
    print("1️⃣  POST /reset — Start a new episode")
    print("   curl -X POST http://localhost:7860/reset \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"task_id\": \"task1\", \"seed\": 42}'")
    print("   Returns: {\"session_id\": \"abc123\", \"observation\": {...}}\n")
    
    print("2️⃣  POST /step — Execute an action")
    print("   curl -X POST http://localhost:7860/step \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{'")
    print("       \"session_id\": \"abc123\",")
    print("       \"action_type\": \"select_supplier\",")
    print("       \"target_id\": \"SUP-1000\",")
    print("       \"parameters\": {\"quantity\": 500},")
    print("       \"explanation\": \"Best cost and reliability balance.\"")
    print("     }'")
    print("   Returns: {\"observation\": {...}, \"reward\": 0.0, \"done\": false}\n")
    
    print("3️⃣  GET /tasks — List all available tasks")
    print("   curl http://localhost:7860/tasks")
    print("   Returns: {\"tasks\": [{...}, {...}, {...}]}\n")
    
    print("4️⃣  GET /docs — Interactive API documentation")
    print("   Open: http://localhost:7860/docs\n")
    
    print("To start the server:")
    print("   uvicorn server:app --host 0.0.0.0 --port 7860\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        demo_api_style()
    else:
        demo_all_tasks()
        demo_api_style()
