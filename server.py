"""
Supply Chain Disruption Gym — OpenEnv Server
FastAPI server implementing the OpenEnv 0.1 spec.
Exposes: POST /reset, POST /step, GET /state, GET /health
WebSocket: /ws  (for streaming RL training)
"""
from __future__ import annotations
import json
import uuid
import asyncio
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import SCAction, ActionType
from environment import SupplyChainEnv


app = FastAPI(
    title="Supply Chain Disruption Gym",
    description=(
        "OpenEnv-spec RL environment for training AI agents to handle "
        "real-world supply chain disruption scenarios. "
        "Built for the Meta/HuggingFace OpenEnv Hackathon 2026."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage: session_id -> SupplyChainEnv
_sessions: Dict[str, SupplyChainEnv] = {}
MAX_SESSIONS = 64


# ─── Request / Response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id:  str = "task1"
    seed:     Optional[int] = None
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id:  str
    action_type: str
    target_id:   Optional[str] = None
    parameters:  dict = {}
    explanation: str  = ""


class StateRequest(BaseModel):
    session_id: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> SupplyChainEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=404,
                            detail=f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


def _parse_action(req: StepRequest) -> SCAction:
    try:
        atype = ActionType(req.action_type)
    except ValueError:
        valid = [a.value for a in ActionType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type '{req.action_type}'. Valid: {valid}")
    return SCAction(
        action_type = atype,
        target_id   = req.target_id,
        parameters  = req.parameters,
        explanation = req.explanation,
    )


# ─── REST endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.post("/reset")
def reset(req: ResetRequest):
    """Start a new episode. Returns initial observation + session_id."""
    if len(_sessions) >= MAX_SESSIONS:
        # Evict oldest session
        oldest = next(iter(_sessions))
        del _sessions[oldest]

    session_id = req.session_id or str(uuid.uuid4())[:8]
    env        = SupplyChainEnv(task_id=req.task_id, seed=req.seed)
    obs        = env.reset(seed=req.seed)
    _sessions[session_id] = env

    return {
        "session_id":  session_id,
        "observation": obs.model_dump(),
        "task_info": {
            "task_id":    req.task_id,
            "description": (
                "task1=Single supplier failure (Easy) | "
                "task2=Port congestion + deadline (Medium) | "
                "task3=Multi-failure + partial info (Hard)"
            ),
            "valid_actions": [a.value for a in ActionType],
        },
    }


@app.post("/step")
def step(req: StepRequest):
    """Execute one agent action. Returns observation, reward, done, info."""
    env    = _get_session(req.session_id)
    action = _parse_action(req)

    obs, reward, done, info = env.step(action)

    response = {
        "observation": obs.model_dump(),
        "reward":      reward,
        "done":        done,
        "info":        {},
    }

    if done and "episode_result" in info:
        result = info["episode_result"]
        response["info"]["episode_result"] = result.model_dump()
        response["info"]["grader_feedback"] = result.grader_feedback
        # Clean up session
        del _sessions[req.session_id]

    return response


@app.get("/state/{session_id}")
def state(session_id: str):
    """Return current state snapshot without advancing the episode."""
    env = _get_session(session_id)
    return env.state().model_dump()


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "task_id":     "task1",
                "name":        "Single Supplier Failure",
                "difficulty":  "Easy",
                "description": (
                    "One primary supplier goes offline. Choose the best "
                    "alternate supplier balancing cost, lead time, and reliability."
                ),
                "max_turns":    5,
                "reward_dims":  ["decision_accuracy", "cost_efficiency",
                                 "explanation_quality", "speed"],
                "baseline_score": 0.65,
            },
            {
                "task_id":     "task2",
                "name":        "Port Congestion + Deadline",
                "difficulty":  "Medium",
                "description": (
                    "JNPT Mumbai congested. 3 shipments affected — one critical pharma. "
                    "Prioritise, reroute/expedite/delay under tight budget."
                ),
                "max_turns":    8,
                "reward_dims":  ["critical_item_handled", "cost_efficiency",
                                 "prioritisation", "explanation_quality"],
                "baseline_score": 0.38,
            },
            {
                "task_id":     "task3",
                "name":        "Multi-Vendor Crisis + Partial Info",
                "difficulty":  "Hard",
                "description": (
                    "2 suppliers offline simultaneously. Inventory data unknown. "
                    "Use request_info tool calls, reason under uncertainty, "
                    "prevent production stoppage."
                ),
                "max_turns":    12,
                "reward_dims":  ["info_gathering", "decision_quality", "cost_efficiency",
                                 "production_continuity", "risk_awareness"],
                "baseline_score": 0.18,
            },
        ]
    }


# ─── WebSocket endpoint (for RL training loops) ───────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket interface for RL training.
    Protocol:
      Client → {"type": "reset", "task_id": "task1", "seed": 42}
      Server ← {"type": "observation", "data": {...}, "session_id": "..."}
      Client → {"type": "step", "session_id": "...", "action_type": "...", ...}
      Server ← {"type": "step_result", "observation": {...}, "reward": 0.0, "done": false}
    """
    await websocket.accept()
    session_id = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "reset":
                task_id    = msg.get("task_id", "task1")
                seed       = msg.get("seed")
                session_id = str(uuid.uuid4())[:8]
                env        = SupplyChainEnv(task_id=task_id, seed=seed)
                obs        = env.reset(seed=seed)
                _sessions[session_id] = env
                await websocket.send_text(json.dumps({
                    "type":       "observation",
                    "session_id": session_id,
                    "data":       obs.model_dump(),
                }))

            elif msg.get("type") == "step":
                if not session_id or session_id not in _sessions:
                    await websocket.send_text(json.dumps(
                        {"type": "error", "message": "No active session. Send reset first."}))
                    continue

                env = _sessions[session_id]
                try:
                    action = SCAction(
                        action_type = ActionType(msg["action_type"]),
                        target_id   = msg.get("target_id"),
                        parameters  = msg.get("parameters", {}),
                        explanation = msg.get("explanation", ""),
                    )
                except (ValueError, KeyError) as e:
                    await websocket.send_text(json.dumps(
                        {"type": "error", "message": str(e)}))
                    continue

                obs, reward, done, info = env.step(action)

                response = {
                    "type":        "step_result",
                    "observation": obs.model_dump(),
                    "reward":      reward,
                    "done":        done,
                }
                if done and "episode_result" in info:
                    response["episode_result"] = info["episode_result"].model_dump()
                    if session_id in _sessions:
                        del _sessions[session_id]

                await websocket.send_text(json.dumps(response))

            elif msg.get("type") == "state":
                if session_id and session_id in _sessions:
                    st = _sessions[session_id].state()
                    await websocket.send_text(json.dumps(
                        {"type": "state", "data": st.model_dump()}))
                else:
                    await websocket.send_text(json.dumps(
                        {"type": "error", "message": "No active session."}))

            else:
                await websocket.send_text(json.dumps(
                    {"type": "error",
                     "message": f"Unknown message type: {msg.get('type')}"}))

    except WebSocketDisconnect:
        if session_id and session_id in _sessions:
            del _sessions[session_id]
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass