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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from models import SCAction, SCObservation, SCState, ActionType
from environment import SupplyChainEnv

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Get the key
api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# Only initialize if the key exists, otherwise use a placeholder or None
if api_key:
    client = OpenAI(api_key=api_key)
else:
    print("WARNING: OPENAI_API_KEY not found. LLM features will be disabled.")
    client = None


app = FastAPI(
    title="GatiAI",
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
async def health_check():
    return {"status": "healthy", "active_sessions": len(_sessions)}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
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


@app.get("/metadata")
def metadata():
    return {
        "name": "GatiAI Logistics",
        "description": "Supply chain crisis management environment for Indian logistics."
    }


@app.get("/schema")
def schema():
    return {
        "action": SCAction.model_json_schema(),
        "observation": SCObservation.model_json_schema(),
        "state": SCState.model_json_schema(),
    }


@app.get("/state")
def get_state():
    return {
        "message": "Use /state/{session_id} with a valid session_id to inspect a live episode state."
    }


@app.post("/mcp")
def mcp():
    return {"jsonrpc": "2.0", "result": "connected"}


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>GatiAI Supply Chain Disruption Gym</title>
            <style>
                body { font-family: Inter, system-ui, sans-serif; margin: 0; padding: 2rem; background: #F7F9FB; color: #111827; }
                .container { max-width: 900px; margin: auto; }
                h1 { color: #0F172A; }
                pre, textarea { background: #111827; color: #f8fafc; padding: 1rem; border-radius: 12px; overflow-x: auto; width: 100%; box-sizing: border-box; border: 1px solid #CBD5E1; }
                .card { background: white; border-radius: 18px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 20px 60px rgba(15,23,42,0.06); }
                button { background: #2563EB; color: white; border: none; border-radius: 10px; padding: 0.9rem 1.4rem; cursor: pointer; font-size: 1rem; }
                button:hover { background: #1D4ED8; }
                .log { width: 100%; min-height: 220px; font-family: monospace; margin-top: 1rem; padding: 1rem; border-radius: 12px; border: 1px solid #CBD5E1; background: #F8FAFC; white-space: pre-wrap; }
                label { display: block; margin-top: 1rem; font-weight: 600; }
                input, select, textarea { width: 100%; margin-top: 0.5rem; padding: 0.8rem; border: 1px solid #CBD5E1; border-radius: 10px; font-size: 0.95rem; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>GatiAI Supply Chain Disruption Gym</h1>
                <p>OpenEnv environment for testing AI agents against supply chain disruption scenarios. Use the built-in UI or API docs for fast experimentation.</p>
                <div class="card">
                    <h2>Quick Start</h2>
                    <button onclick="resetTask()">Reset task1</button>
                    <button onclick="refreshState()" style="margin-left: 1rem;">Refresh state</button>
                    <div id="log" class="log">Ready. Click reset to begin.</div>
                </div>
                <div class="card">
                    <h2>Agent Action</h2>
                    <label for="actionType">Action Type</label>
                    <select id="actionType">
                        <option value="select_supplier">select_supplier</option>
                        <option value="reroute_shipment">reroute_shipment</option>
                        <option value="expedite_shipment">expedite_shipment</option>
                        <option value="delay_order">delay_order</option>
                        <option value="request_info">request_info</option>
                        <option value="escalate_to_human">escalate_to_human</option>
                        <option value="split_order">split_order</option>
                    </select>
                    <label for="targetId">Target ID</label>
                    <input id="targetId" placeholder="supplier_id or shipment_id" />
                    <label for="parameters">Parameters (JSON)</label>
                    <textarea id="parameters" rows="4" placeholder='{"quantity": 500, "delay_days": 7}'></textarea>
                    <label for="explanation">Explanation</label>
                    <textarea id="explanation" rows="4" placeholder="Explain your decision in plain language."></textarea>
                    <button onclick="submitAction()" style="margin-top: 1rem;">Submit action</button>
                </div>
                <div class="card">
                    <h2>API Reference</h2>
                    <pre>POST /reset
POST /step
GET /tasks
GET /health
GET /docs</pre>
                </div>
            </div>
            <script>
                const logEl = document.getElementById('log');
                let currentSession = null;

                function appendLog(message) {
                    logEl.textContent = message;
                }

                async function resetTask() {
                    appendLog('Sending reset...');
                    const response = await fetch('/reset', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ task_id: 'task1', seed: 42 })
                    });
                    const data = await response.json();
                    currentSession = data.session_id;
                    appendLog('Session started: ' + currentSession + '\n\n' + JSON.stringify(data, null, 2));
                }

                async function refreshState() {
                    if (!currentSession) {
                        appendLog('No active session. Reset first.');
                        return;
                    }
                    const response = await fetch(`/state/${currentSession}`);
                    const data = await response.json();
                    appendLog('Current state:\n' + JSON.stringify(data, null, 2));
                }

                async function submitAction() {
                    if (!currentSession) {
                        appendLog('No active session. Reset first.');
                        return;
                    }
                    const actionType = document.getElementById('actionType').value;
                    const targetId = document.getElementById('targetId').value || undefined;
                    const paramsText = document.getElementById('parameters').value.trim();
                    const explanation = document.getElementById('explanation').value.trim();
                    let parameters = {};
                    if (paramsText) {
                        try {
                            parameters = JSON.parse(paramsText);
                        } catch (err) {
                            appendLog('Invalid JSON in parameters: ' + err.message);
                            return;
                        }
                    }
                    const response = await fetch('/step', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: currentSession,
                            action_type: actionType,
                            target_id: targetId,
                            parameters: parameters,
                            explanation: explanation,
                        })
                    });
                    const data = await response.json();
                    appendLog('Step result:\n' + JSON.stringify(data, null, 2));
                }
            </script>
        </body>
        </html>
        """
    )


@app.post("/predict")
def llm_baseline(state: dict):
    if client is None:
        return {"error": "OpenAI API key not configured"}
    
    # 1. Prepare the Prompt
    prompt = f"""
    You are an AI Logistics Manager for GatiAI Supply Chain Disruption Gym.
    Current Environment State: {json.dumps(state, indent=2)}
    
    Available Actions:
    - select_supplier: Choose an alternate supplier by supplier_id
    - reroute_shipment: Reroute a delayed shipment
    - delay_order: Accept delay penalty on a low-urgency order
    - request_info: Request hidden information
    - escalate_to_human: Hand off to human supervisor
    - split_order: Split order across multiple suppliers
    - expedite_shipment: Pay premium for fast delivery
    
    Choose the best action_type from the list above to handle the current disruption.
    Return ONLY the action_type as a string, e.g., "select_supplier"
    """

    # 2. Call OpenAI
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # 3. Extract Action
    action_type = response.choices[0].message.content.strip()
    return {"action_type": action_type}


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