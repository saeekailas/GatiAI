"""
Supply Chain Disruption Gym — Scenario Generator
Generates realistic synthetic Indian supply chain crises.
All data is 100% synthetic — no real company information.
"""
from __future__ import annotations
import random
from typing import List, Tuple
from models import (
    Supplier, Shipment, WarehouseStock,
    DisruptionType, UrgencyLevel
)


# ─── Indian geography constants (synthetic) ───────────────────────────────────

INDIAN_CITIES   = ["Mumbai", "Delhi", "Pune", "Chennai", "Hyderabad",
                   "Ahmedabad", "Bengaluru", "Kolkata", "Surat", "Jaipur"]
INDIAN_PORTS    = ["JNPT Mumbai", "Mundra Port", "Chennai Port",
                   "Vizag Port", "Kolkata Port"]
MATERIALS       = ["Steel sheets", "Electronic components", "Pharma APIs",
                   "Textile yarn", "Auto parts", "Chemical reagents",
                   "Packaging material", "Semiconductor chips"]
CARGO_TYPES     = ["Pharmaceuticals", "Auto components", "Consumer electronics",
                   "Textiles", "Industrial chemicals", "Food products"]


def _make_suppliers(n: int = 5, available_count: int = None,
                    seed: int = None) -> List[Supplier]:
    """Generate n synthetic suppliers, some potentially unavailable."""
    rng = random.Random(seed)
    if available_count is None:
        available_count = n

    suppliers = []
    for i in range(n):
        city = rng.choice(INDIAN_CITIES)
        suppliers.append(Supplier(
            supplier_id       = f"SUP-{1000 + i}",
            name              = f"IndSupplier_{city}_{i+1}",
            location          = city,
            cost_per_unit     = round(rng.uniform(80, 500), 2),
            lead_time_days    = rng.randint(2, 14),
            reliability_score = round(rng.uniform(0.55, 0.98), 2),
            available         = (i < available_count),
            current_stock     = rng.randint(500, 5000) if i < available_count else 0,
        ))
    return suppliers


def _make_shipments(n: int = 3, seed: int = None) -> List[Shipment]:
    rng = random.Random(seed)
    shipments = []
    for i in range(n):
        origin      = rng.choice(INDIAN_PORTS)
        destination = rng.choice(INDIAN_CITIES)
        cargo       = rng.choice(CARGO_TYPES)
        urgency     = rng.choice(list(UrgencyLevel))
        shipments.append(Shipment(
            shipment_id      = f"SHP-{2000 + i}",
            origin           = origin,
            destination      = destination,
            cargo            = cargo,
            quantity         = rng.randint(100, 2000),
            deadline_days    = rng.randint(2, 10),
            urgency          = urgency,
            current_status   = "In transit — disrupted",
            cost_to_reroute  = round(rng.uniform(5000, 50000), 2),
            cost_to_expedite = round(rng.uniform(8000, 80000), 2),
            delay_penalty    = round(rng.uniform(2000, 20000), 2),
        ))
    return shipments


def _make_stocks(seed: int = None) -> List[WarehouseStock]:
    rng = random.Random(seed)
    stocks = []
    for mat in rng.sample(MATERIALS, k=3):
        current = rng.randint(50, 2000)
        min_s   = rng.randint(200, 500)
        daily   = rng.randint(30, 150)
        stocks.append(WarehouseStock(
            material          = mat,
            current_stock     = current,
            minimum_stock     = min_s,
            daily_consumption = daily,
            days_remaining    = max(0, (current - min_s) // daily),
        ))
    return stocks


# ─── Public scenario builders ─────────────────────────────────────────────────

def build_task1_scenario(seed: int = 42) -> dict:
    """
    Task 1 — Easy: Single supplier failure.
    One main supplier is down. 3 alternates available with clear trade-offs.
    """
    main = _make_suppliers(n=1, available_count=0, seed=seed)[0]
    main.name = f"PrimarySupplier_{main.location}"

    alts = _make_suppliers(n=3, available_count=3, seed=seed + 10)
    # Ensure one clearly best option
    alts[0].cost_per_unit     = 120.0
    alts[0].lead_time_days    = 3
    alts[0].reliability_score = 0.95
    alts[1].cost_per_unit     = 95.0
    alts[1].lead_time_days    = 8
    alts[1].reliability_score = 0.72
    alts[2].cost_per_unit     = 200.0
    alts[2].lead_time_days    = 2
    alts[2].reliability_score = 0.98

    stocks = _make_stocks(seed=seed)
    # Make one material critically low
    stocks[0].current_stock   = 80
    stocks[0].minimum_stock   = 200
    stocks[0].days_remaining  = 0

    return {
        "disruption_type":        DisruptionType.SUPPLIER_FAILURE,
        "disruption_description": (
            f"ALERT: {main.name} ({main.location}) has gone offline due to "
            "a factory fire. Production halted. Your main raw material supply "
            "for Steel sheets is now at risk. Current warehouse stock will run "
            "out in less than 1 day. Select an alternate supplier immediately."
        ),
        "available_suppliers":    alts,
        "affected_shipments":     [],
        "warehouse_stocks":       stocks,
        "budget_remaining":       500000.0,
        "time_remaining_days":    5,
        "partial_info":           False,
        "optimal_action":         "select_supplier",
        "optimal_target":         alts[0].supplier_id,
        "optimal_cost":           alts[0].cost_per_unit * 500,
        "max_turns":              5,
    }


def build_task2_scenario(seed: int = 99) -> dict:
    """
    Task 2 — Medium: Port congestion + deadline pressure.
    Mumbai JNPT congested. 3 shipments affected — one is critical pharma.
    Agent must prioritise, reroute, work under budget constraint.
    """
    shipments = _make_shipments(n=3, seed=seed)
    shipments[0].cargo      = "Pharmaceuticals"
    shipments[0].urgency    = UrgencyLevel.CRITICAL
    shipments[0].deadline_days = 2
    shipments[0].destination = "Hyderabad"

    shipments[1].cargo      = "Auto components"
    shipments[1].urgency    = UrgencyLevel.MEDIUM
    shipments[1].deadline_days = 7

    shipments[2].cargo      = "Textiles"
    shipments[2].urgency    = UrgencyLevel.LOW
    shipments[2].deadline_days = 14

    alts = _make_suppliers(n=2, available_count=2, seed=seed + 5)
    stocks = _make_stocks(seed=seed)

    return {
        "disruption_type":        DisruptionType.PORT_CONGESTION,
        "disruption_description": (
            "ALERT: JNPT Mumbai is experiencing severe congestion — "
            "estimated 5-day delay for all container clearances. "
            "3 of your shipments are affected. Budget for rerouting/expediting "
            "is limited to ₹80,000. The pharma shipment has a hard 2-day deadline "
            "— missing it means patient care disruption and heavy penalties. "
            "Prioritise and decide actions for each shipment."
        ),
        "available_suppliers":    alts,
        "affected_shipments":     shipments,
        "warehouse_stocks":       stocks,
        "budget_remaining":       80000.0,
        "time_remaining_days":    2,
        "partial_info":           False,
        "optimal_actions": [
            {"target": shipments[0].shipment_id, "action": "expedite_shipment"},
            {"target": shipments[1].shipment_id, "action": "reroute_shipment"},
            {"target": shipments[2].shipment_id, "action": "delay_order"},
        ],
        "max_turns":              8,
    }


def build_task3_scenario(seed: int = 7) -> dict:
    """
    Task 3 — Hard: Multi-vendor failure + partial information.
    2 vendors offline simultaneously. Agent doesn't know full inventory.
    Must use request_info tool calls, handle uncertainty, minimise stoppage.
    """
    rng = random.Random(seed)

    suppliers = _make_suppliers(n=5, available_count=2, seed=seed)
    # First 3 unavailable (2 failed + 1 unreliable flagged)
    for s in suppliers[:3]:
        s.available = False
        s.current_stock = 0

    shipments = _make_shipments(n=2, seed=seed)
    stocks = _make_stocks(seed=seed)

    # Hide stock data to simulate partial info
    for s in stocks:
        s.current_stock = -1   # -1 = unknown, agent must request_info

    hidden_info = {
        "stock_query_steel":     {"material": "Steel sheets",     "actual_stock": rng.randint(100, 800)},
        "stock_query_pharma":    {"material": "Pharma APIs",       "actual_stock": rng.randint(20, 150)},
        "vendor_status_query":   {"vendor": suppliers[3].name,    "status": "available", "lead_time": 4},
    }

    return {
        "disruption_type":        DisruptionType.MULTI_FAILURE,
        "disruption_description": (
            "CRITICAL: SUP-1000 and SUP-1001 have both gone offline — "
            "SUP-1000 due to labour strike, SUP-1001 due to regulatory shutdown. "
            "A third supplier (SUP-1002) has been flagged as unreliable this week. "
            "Exact warehouse inventory is UNKNOWN — sensors offline. "
            "You have 2 remaining suppliers but don't know their current capacity. "
            "Use request_info to gather what you need before committing to decisions. "
            "Production stoppage penalty: ₹200,000/day. Budget: ₹150,000."
        ),
        "available_suppliers":    suppliers,
        "affected_shipments":     shipments,
        "warehouse_stocks":       stocks,
        "budget_remaining":       150000.0,
        "time_remaining_days":    3,
        "partial_info":           True,
        "hidden_info":            hidden_info,
        "max_turns":              12,
        "production_stoppage_penalty": 200000.0,
    }


def get_scenario(task_id: str, seed: int = None) -> dict:
    """Entry point — returns scenario dict for a given task."""
    _seed = seed if seed is not None else random.randint(1, 9999)
    if task_id == "task1":
        return build_task1_scenario(seed=_seed)
    elif task_id == "task2":
        return build_task2_scenario(seed=_seed)
    elif task_id == "task3":
        return build_task3_scenario(seed=_seed)
    else:
        raise ValueError(f"Unknown task_id: {task_id}. Choose task1, task2, or task3.")