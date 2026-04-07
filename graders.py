"""
Supply Chain Disruption Gym — Graders
Novel multi-dimensional grading: correctness + cost efficiency + speed + risk awareness.
This is the research contribution — no other OpenEnv environment grades cost efficiency.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
from models import SCAction, ActionType, EpisodeResult


# ─── Base grader ──────────────────────────────────────────────────────────────

class BaseGrader:
    """All graders inherit from this. Returns reward 0.0–1.0."""

    def grade(
        self,
        actions:       List[SCAction],
        scenario:      dict,
        final_state:   dict,
    ) -> EpisodeResult:
        raise NotImplementedError


# ─── Explanation quality helper ───────────────────────────────────────────────

def _grade_explanation(explanation: str) -> float:
    """
    Simple heuristic explanation grader.
    Full LLM-based grading can replace this in production.
    """
    if not explanation or len(explanation) < 10:
        return 0.0
    score = 0.0
    explanation_lower = explanation.lower()
    # Reasoning indicators
    for kw in ["because", "due to", "since", "cost", "risk", "deadline",
               "cheaper", "faster", "reliable", "penalty", "urgent", "stock"]:
        if kw in explanation_lower:
            score += 0.1
    # Length bonus (more reasoned explanation)
    if len(explanation) > 50:
        score += 0.1
    if len(explanation) > 120:
        score += 0.1
    return min(score, 1.0)


# ─── Task 1 Grader ─────────────────────────────────────────────────────────────

class Task1Grader(BaseGrader):
    """
    Task 1: Single supplier failure.
    Grades: correct supplier selected + cost efficiency + explanation quality.
    Optimal = alts[0]: cost 120, lead 3 days, reliability 0.95
    """

    def grade(self, actions, scenario, final_state) -> EpisodeResult:
        suppliers   = {s.supplier_id: s for s in scenario["available_suppliers"]}
        optimal_id  = scenario["optimal_target"]
        optimal_cost = scenario["optimal_cost"]
        budget      = scenario["budget_remaining"]

        select_actions = [a for a in actions if a.action_type == ActionType.SELECT_SUPPLIER]
        escalated      = any(a.action_type == ActionType.ESCALATE_TO_HUMAN for a in actions)

        scores: Dict[str, float] = {
            "decision_accuracy":    0.0,
            "cost_efficiency":      0.0,
            "explanation_quality":  0.0,
            "speed":                0.0,
        }
        feedback_parts = []

        # 1. Decision accuracy (0.35)
        if select_actions:
            chosen_id   = select_actions[-1].target_id
            chosen      = suppliers.get(chosen_id)
            if chosen_id == optimal_id:
                scores["decision_accuracy"] = 0.35
                feedback_parts.append("Correct supplier selected (optimal cost + reliability).")
            elif chosen and chosen.available:
                # Partial credit for any valid available supplier
                scores["decision_accuracy"] = 0.15
                feedback_parts.append(f"Valid but suboptimal supplier selected ({chosen.name}).")
            else:
                feedback_parts.append("Selected supplier unavailable or invalid.")
        elif escalated:
            scores["decision_accuracy"] = 0.10
            feedback_parts.append("Escalated to human without attempting resolution.")
        else:
            feedback_parts.append("No supplier selected — production remains halted.")

        # 2. Cost efficiency (0.30)
        if select_actions:
            chosen_id = select_actions[-1].target_id
            chosen    = suppliers.get(chosen_id)
            if chosen and chosen.available:
                chosen_cost = chosen.cost_per_unit * 500  # standard order qty
                if chosen_cost <= optimal_cost * 1.05:
                    scores["cost_efficiency"] = 0.30
                    feedback_parts.append("Cost-efficient decision — within 5% of optimal.")
                elif chosen_cost <= optimal_cost * 1.30:
                    scores["cost_efficiency"] = 0.18
                    feedback_parts.append("Acceptable cost but 10–30% above optimal.")
                else:
                    scores["cost_efficiency"] = 0.05
                    feedback_parts.append("Costly decision — significantly above optimal price.")

        # 3. Explanation quality (0.25)
        all_explanations = " ".join(a.explanation for a in actions)
        scores["explanation_quality"] = _grade_explanation(all_explanations) * 0.25
        if scores["explanation_quality"] > 0.15:
            feedback_parts.append("Good reasoning provided in explanation.")
        else:
            feedback_parts.append("Explanation lacks sufficient reasoning.")

        # 4. Speed (0.10) — fewer turns = higher score
        turns_taken = final_state.get("turns_taken", len(actions))
        max_turns   = scenario["max_turns"]
        turn_ratio  = turns_taken / max(max_turns, 1)
        scores["speed"] = round((1.0 - turn_ratio) * 0.10, 3)

        total_reward = round(sum(scores.values()), 3)
        success      = total_reward >= 0.60

        return EpisodeResult(
            task_id             = "task1",
            success             = success,
            final_reward        = total_reward,
            score_breakdown     = scores,
            total_cost          = final_state.get("total_cost", 0),
            turns_taken         = turns_taken,
            production_stopped  = final_state.get("production_stopped", True),
            explanation_quality = scores["explanation_quality"],
            grader_feedback     = " | ".join(feedback_parts),
        )


# ─── Task 2 Grader ─────────────────────────────────────────────────────────────

class Task2Grader(BaseGrader):
    """
    Task 2: Port congestion + deadline.
    Grades: correct prioritisation + budget adherence + cost efficiency + explanation.
    Critical pharma shipment MUST be expedited — this is non-negotiable.
    """

    def grade(self, actions, scenario, final_state) -> EpisodeResult:
        shipments      = {s.shipment_id: s for s in scenario["affected_shipments"]}
        budget         = scenario["budget_remaining"]
        optimal_actions = scenario["optimal_actions"]   # list of {target, action}

        scores: Dict[str, float] = {
            "critical_item_handled": 0.0,
            "cost_efficiency":       0.0,
            "prioritisation":        0.0,
            "explanation_quality":   0.0,
        }
        feedback_parts = []
        total_cost     = 0.0

        # Find pharma shipment (critical)
        pharma_shp  = next((s for s in scenario["affected_shipments"]
                            if s.cargo == "Pharmaceuticals"), None)

        # 1. Critical item handled (0.35) — pharma MUST be expedited
        pharma_actions = [a for a in actions
                          if a.target_id == (pharma_shp.shipment_id if pharma_shp else None)]
        if any(a.action_type == ActionType.EXPEDITE_SHIPMENT for a in pharma_actions):
            scores["critical_item_handled"] = 0.35
            feedback_parts.append("Critical pharma shipment correctly expedited.")
            total_cost += pharma_shp.cost_to_expedite if pharma_shp else 0
        elif any(a.action_type == ActionType.REROUTE_SHIPMENT for a in pharma_actions):
            scores["critical_item_handled"] = 0.20
            feedback_parts.append("Pharma shipment rerouted — acceptable but expediting preferred.")
            total_cost += pharma_shp.cost_to_reroute if pharma_shp else 0
        else:
            feedback_parts.append("CRITICAL FAIL: Pharma shipment not prioritised — patient care at risk.")

        # 2. Budget adherence + cost efficiency (0.30)
        for a in actions:
            shp = shipments.get(a.target_id)
            if not shp:
                continue
            if a.action_type == ActionType.EXPEDITE_SHIPMENT:
                total_cost += shp.cost_to_expedite
            elif a.action_type == ActionType.REROUTE_SHIPMENT:
                total_cost += shp.cost_to_reroute

        if total_cost <= budget:
            # Cost efficiency: closer to budget = better use of resources
            efficiency = 1.0 - (total_cost / budget)
            # Reward staying under budget while handling all items
            if efficiency < 0.6:  # used >40% of budget
                scores["cost_efficiency"] = 0.30
                feedback_parts.append(f"Good budget utilisation: ₹{total_cost:,.0f} / ₹{budget:,.0f}.")
            else:
                scores["cost_efficiency"] = 0.20
                feedback_parts.append("Budget underspent — some shipments may be under-served.")
        else:
            scores["cost_efficiency"] = 0.0
            feedback_parts.append(f"BUDGET EXCEEDED: ₹{total_cost:,.0f} vs limit ₹{budget:,.0f}.")

        # 3. Prioritisation logic (0.25) — low-urgency delayed, medium rerouted
        low_shp    = next((s for s in scenario["affected_shipments"]
                           if s.urgency.value == "low"), None)
        medium_shp = next((s for s in scenario["affected_shipments"]
                           if s.urgency.value == "medium"), None)

        prioritisation_score = 0.0
        if low_shp:
            low_actions = [a for a in actions if a.target_id == low_shp.shipment_id]
            if any(a.action_type == ActionType.DELAY_ORDER for a in low_actions):
                prioritisation_score += 0.12
                feedback_parts.append("Low-urgency shipment correctly delayed.")
        if medium_shp:
            med_actions = [a for a in actions if a.target_id == medium_shp.shipment_id]
            if any(a.action_type in (ActionType.REROUTE_SHIPMENT, ActionType.EXPEDITE_SHIPMENT)
                   for a in med_actions):
                prioritisation_score += 0.13
                feedback_parts.append("Medium-urgency shipment handled appropriately.")
        scores["prioritisation"] = prioritisation_score

        # 4. Explanation quality (0.10)
        all_expl = " ".join(a.explanation for a in actions)
        scores["explanation_quality"] = _grade_explanation(all_expl) * 0.10

        total_reward = round(sum(scores.values()), 3)
        success      = total_reward >= 0.55 and scores["critical_item_handled"] > 0

        return EpisodeResult(
            task_id             = "task2",
            success             = success,
            final_reward        = total_reward,
            score_breakdown     = scores,
            total_cost          = total_cost,
            turns_taken         = final_state.get("turns_taken", len(actions)),
            production_stopped  = final_state.get("production_stopped", False),
            explanation_quality = scores["explanation_quality"],
            grader_feedback     = " | ".join(feedback_parts),
        )


# ─── Task 3 Grader ─────────────────────────────────────────────────────────────

class Task3Grader(BaseGrader):
    """
    Task 3: Multi-failure + partial information.
    Grades: information gathering quality + decision under uncertainty
            + cost + production continuity + risk escalation.
    This is the hardest task — expected baseline score ~0.18.
    """

    def grade(self, actions, scenario, final_state) -> EpisodeResult:
        suppliers   = scenario["available_suppliers"]
        budget      = scenario["budget_remaining"]
        stoppage_penalty = scenario.get("production_stoppage_penalty", 200000.0)

        scores: Dict[str, float] = {
            "info_gathering":        0.0,
            "decision_quality":      0.0,
            "cost_efficiency":       0.0,
            "production_continuity": 0.0,
            "risk_awareness":        0.0,
        }
        feedback_parts = []

        # 1. Info gathering (0.20) — agent should call request_info before deciding
        info_requests = [a for a in actions if a.action_type == ActionType.REQUEST_INFO]
        select_actions = [a for a in actions if a.action_type == ActionType.SELECT_SUPPLIER]

        if info_requests:
            # Reward gathering info BEFORE committing to supplier
            first_select_turn  = next(
                (i for i, a in enumerate(actions)
                 if a.action_type == ActionType.SELECT_SUPPLIER), len(actions))
            info_before_select = sum(
                1 for i, a in enumerate(actions)
                if a.action_type == ActionType.REQUEST_INFO and i < first_select_turn)

            if info_before_select >= 2:
                scores["info_gathering"] = 0.20
                feedback_parts.append(f"Excellent: {info_before_select} info requests made before deciding.")
            elif info_before_select == 1:
                scores["info_gathering"] = 0.12
                feedback_parts.append("One info request made — more recommended under uncertainty.")
            else:
                scores["info_gathering"] = 0.05
                feedback_parts.append("Info requested after decisions — poor ordering under uncertainty.")
        else:
            feedback_parts.append("No information requested — agent decided with unknown inventory.")

        # 2. Decision quality (0.25) — chose from actually available suppliers
        available_ids = {s.supplier_id for s in suppliers if s.available}
        if select_actions:
            valid_selections = [a for a in select_actions if a.target_id in available_ids]
            if valid_selections:
                scores["decision_quality"] = 0.25
                feedback_parts.append("Selected from available suppliers correctly.")
            else:
                feedback_parts.append("Selected unavailable/failed supplier — critical error.")
        else:
            escalated = any(a.action_type == ActionType.ESCALATE_TO_HUMAN for a in actions)
            if escalated:
                scores["decision_quality"] = 0.12
                feedback_parts.append("Escalated to human — appropriate under high uncertainty.")
            else:
                feedback_parts.append("No action taken — production will halt.")

        # 3. Cost efficiency (0.20)
        total_cost = final_state.get("total_cost", 0)
        if total_cost <= budget:
            cost_ratio = total_cost / max(budget, 1)
            if cost_ratio > 0.3:
                scores["cost_efficiency"] = 0.20
                feedback_parts.append(f"Budget used efficiently: ₹{total_cost:,.0f}.")
            else:
                scores["cost_efficiency"] = 0.10
                feedback_parts.append("Underspent — may not have addressed all disruptions.")
        else:
            scores["cost_efficiency"] = 0.0
            feedback_parts.append(f"Over budget by ₹{total_cost - budget:,.0f}.")

        # 4. Production continuity (0.20) — did agent prevent stoppage?
        production_stopped = final_state.get("production_stopped", True)
        if not production_stopped:
            scores["production_continuity"] = 0.20
            feedback_parts.append("Production continuity maintained — excellent outcome.")
        else:
            # Partial credit if agent reduced stoppage duration
            actions_toward_continuity = len([
                a for a in actions
                if a.action_type in (ActionType.SELECT_SUPPLIER,
                                     ActionType.EXPEDITE_SHIPMENT,
                                     ActionType.SPLIT_ORDER)
            ])
            if actions_toward_continuity > 0:
                scores["production_continuity"] = 0.08
                feedback_parts.append("Production stopped but agent attempted mitigation.")
            else:
                feedback_parts.append("Production halted — no mitigation actions taken.")

        # 5. Risk awareness (0.15) — did agent flag risky decisions?
        escalated = any(a.action_type == ActionType.ESCALATE_TO_HUMAN for a in actions)
        all_expl  = " ".join(a.explanation for a in actions).lower()
        risk_keywords = ["uncertain", "risk", "unknown", "partial", "verify",
                         "confirm", "caution", "escalat", "human review"]
        risk_mentions = sum(1 for kw in risk_keywords if kw in all_expl)

        if escalated or risk_mentions >= 3:
            scores["risk_awareness"] = 0.15
            feedback_parts.append("Good risk awareness shown in decisions/explanations.")
        elif risk_mentions >= 1:
            scores["risk_awareness"] = 0.08
            feedback_parts.append("Some risk acknowledgement — more explicit reasoning helpful.")
        else:
            feedback_parts.append("No risk awareness shown — agent acted without flagging uncertainty.")

        total_reward = round(sum(scores.values()), 3)
        success      = total_reward >= 0.50

        return EpisodeResult(
            task_id             = "task3",
            success             = success,
            final_reward        = total_reward,
            score_breakdown     = scores,
            total_cost          = total_cost,
            turns_taken         = final_state.get("turns_taken", len(actions)),
            production_stopped  = production_stopped,
            explanation_quality = _grade_explanation(" ".join(a.explanation for a in actions)),
            grader_feedback     = " | ".join(feedback_parts),
        )


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_grader(task_id: str) -> BaseGrader:
    mapping = {
        "task1": Task1Grader(),
        "task2": Task2Grader(),
        "task3": Task3Grader(),
    }
    if task_id not in mapping:
        raise ValueError(f"Unknown task_id: {task_id}")
    return mapping[task_id]