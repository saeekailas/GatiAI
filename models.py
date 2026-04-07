"""
Supply Chain Disruption Gym — Typed Models
OpenEnv spec: typed actions, observations, state
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class DisruptionType(str, Enum):
    SUPPLIER_FAILURE   = "supplier_failure"
    PORT_CONGESTION    = "port_congestion"
    WEATHER_DELAY      = "weather_delay"
    CUSTOMS_HOLD       = "customs_hold"
    VEHICLE_BREAKDOWN  = "vehicle_breakdown"
    MULTI_FAILURE      = "multi_failure"


class UrgencyLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    SELECT_SUPPLIER      = "select_supplier"
    REROUTE_SHIPMENT     = "reroute_shipment"
    DELAY_ORDER          = "delay_order"
    REQUEST_INFO         = "request_info"
    ESCALATE_TO_HUMAN    = "escalate_to_human"
    SPLIT_ORDER          = "split_order"
    EXPEDITE_SHIPMENT    = "expedite_shipment"


# ─── Supplier / Shipment data ──────────────────────────────────────────────────

class Supplier(BaseModel):
    supplier_id:      str
    name:             str
    location:         str
    cost_per_unit:    float
    lead_time_days:   int
    reliability_score: float   # 0.0–1.0
    available:        bool = True
    current_stock:    Optional[int] = None


class Shipment(BaseModel):
    shipment_id:     str
    origin:          str
    destination:     str
    cargo:           str
    quantity:        int
    deadline_days:   int
    urgency:         UrgencyLevel
    current_status:  str
    cost_to_reroute: float
    cost_to_expedite: float
    delay_penalty:   float       # cost per day late


class WarehouseStock(BaseModel):
    material:        str
    current_stock:   int
    minimum_stock:   int
    daily_consumption: int
    days_remaining:  int         # derived


# ─── Action ───────────────────────────────────────────────────────────────────

class SCAction(BaseModel):
    action_type:  ActionType
    target_id:    Optional[str]   = None   # supplier_id, shipment_id, etc.
    parameters:   Dict[str, Any]  = Field(default_factory=dict)
    explanation:  str             = ""     # agent must explain its reasoning


# ─── Observation ──────────────────────────────────────────────────────────────

class SCObservation(BaseModel):
    turn:               int
    task_id:            str
    disruption_type:    DisruptionType
    disruption_description: str
    available_suppliers: List[Supplier]
    affected_shipments:  List[Shipment]
    warehouse_stocks:    List[WarehouseStock]
    budget_remaining:    float
    time_remaining_days: int
    partial_info:        bool              # True = some data hidden (Task 3)
    info_available:      Dict[str, Any]    # extra context revealed via request_info
    last_action_result:  Optional[str]     = None
    done:                bool              = False
    message:             str              = ""


# ─── State ────────────────────────────────────────────────────────────────────

class SCState(BaseModel):
    task_id:             str
    turn:                int
    max_turns:           int
    disruption_type:     DisruptionType
    total_cost_incurred: float
    production_stopped:  bool
    orders_delayed:      int
    escalated:           bool
    actions_taken:       List[Dict[str, Any]]
    current_score:       float
    done:                bool


# ─── Episode Result ───────────────────────────────────────────────────────────

class EpisodeResult(BaseModel):
    task_id:                str
    success:                bool
    final_reward:           float
    score_breakdown:        Dict[str, float]
    total_cost:             float
    turns_taken:            int
    production_stopped:     bool
    explanation_quality:    float
    grader_feedback:        str