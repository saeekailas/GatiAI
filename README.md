# Supply Chain Disruption Gym 🏭

**An OpenEnv RL environment for training AI agents to handle real-world Indian supply chain crises.**

Built for the **Meta/HuggingFace OpenEnv Hackathon 2026** | Statement 2: Package using OpenEnv for automated evaluation.

---

## What This Solves

India spends **14–16% of GDP on logistics** vs 8–10% in developed countries. When disruptions hit — supplier failures, port congestion, multi-vendor crises — most companies still respond via phone calls and spreadsheets. Response time: days. Should be: minutes.

This environment trains AI agents to reason through supply chain crises: assess the situation, request missing information, evaluate cost/speed/risk trade-offs, and take the optimal action — automatically.

**No existing OpenEnv environment covers this domain.**

---

## Environment Overview

| Task | Name | Difficulty | Max Turns | Baseline Score |
|------|------|-----------|-----------|----------------|
| `task1` | Single Supplier Failure | Easy | 5 | 0.65 |
| `task2` | Port Congestion + Deadline | Medium | 8 | 0.38 |
| `task3` | Multi-Vendor Crisis + Partial Info | Hard | 12 | 0.18 |

### Novel Reward Function
Unlike other environments that only grade *correctness*, this environment grades:
- **Decision accuracy** — was the right supplier/action chosen?
- **Cost efficiency** — was it the cheapest valid solution?
- **Speed** — fewer turns = higher score
- **Risk awareness** — did the agent flag uncertainty appropriately?
- **Explanation quality** — did the agent reason transparently?

---

## Quick Start

### Install & Run Locally

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/GatiAI
cd supply-chain-gym
pip install -r requirements.txt

# Run the server
uvicorn server:app --host 0.0.0.0 --port 7860

# Run baseline agent (reproduces published scores)
python baseline.py
```

### Docker

```bash
docker build -t GatiAI
docker run -p 7860:7860 GatiAI
```

### Hugging Face Spaces

This project is a REST-first OpenEnv environment with an interactive homepage at `/` and full API docs at `/docs`.

If you deploy with Docker, make sure the Space uses port `7860`, and the container starts with:

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

Open the Space URL in your browser to see the interactive environment landing page.

---

## Testing

Run the test suite locally to verify the environment, API, and episode flow:

```bash
pytest
```

The test suite includes:
- `tests/test_server.py` — API route and schema coverage
- `tests/test_environment.py` — reset/step/state behavior for key tasks

---

## API Reference

### Reset — start a new episode

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1", "seed": 42}'
```

Response includes `session_id` and initial `observation`.
### Agent Evaluation Harness

This project includes a reusable evaluation harness in `agent_evaluator.py` that runs any agent across all tasks, multiple random seeds, and returns a rich summary of performance.

```bash
python3 agent_evaluator.py
```

The evaluator reports:
- `average_reward` per task and aggregate
- `success_rate` per task and aggregate
- full `episode_result` breakdown for every run
- action history for each episode

Use `RuleBasedAgent` from `baseline.py` as a reference implementation for best-practice testing.
### Step — execute one action

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "action_type": "select_supplier",
    "target_id": "SUP-1010",
    "parameters": {"quantity": 500},
    "explanation": "Selected because lowest cost with acceptable lead time and high reliability score of 0.95."
  }'
```

### State — get current snapshot

```bash
curl http://localhost:7860/state/abc123
```

### Valid Action Types

| Action | Description |
|--------|-------------|
| `select_supplier` | Choose alternate supplier by `target_id` |
| `reroute_shipment` | Reroute delayed shipment |
| `expedite_shipment` | Pay premium for fast delivery |
| `delay_order` | Accept delay on low-urgency orders |
| `request_info` | Reveal hidden data (Task 3) |
| `escalate_to_human` | Hand off to human supervisor |
| `split_order` | Distribute order across suppliers |

---

## Observation Space

```python
{
  "turn": int,
  "task_id": str,
  "disruption_type": "supplier_failure | port_congestion | multi_failure | ...",
  "disruption_description": str,        # full crisis description in plain English
  "available_suppliers": [
    {
      "supplier_id": str,
      "name": str,
      "location": str,                  # Indian city
      "cost_per_unit": float,           # INR
      "lead_time_days": int,
      "reliability_score": float,       # 0.0–1.0
      "available": bool
    }
  ],
  "affected_shipments": [
    {
      "shipment_id": str,
      "origin": str,                    # Indian port
      "destination": str,
      "cargo": str,
      "urgency": "low | medium | high | critical",
      "deadline_days": int,
      "cost_to_reroute": float,
      "cost_to_expedite": float,
      "delay_penalty": float            # per day
    }
  ],
  "warehouse_stocks": [
    {
      "material": str,
      "current_stock": int,             # -1 = unknown (Task 3)
      "minimum_stock": int,
      "days_remaining": int
    }
  ],
  "budget_remaining": float,            # INR
  "time_remaining_days": int,
  "partial_info": bool,                 # True in Task 3
  "info_available": dict,               # revealed via request_info
  "last_action_result": str
}
```

---

## Training with TRL + GRPO

```python
from trl import GRPOTrainer, GRPOConfig
import openenv

# Connect to the environment
env_client = openenv.from_hub("YOUR_USERNAME/GatiA")

config = GRPOConfig(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    environment=env_client,
    task_id="task1",   # start with easy task
    num_episodes=1000,
)

trainer = GRPOTrainer(config)
trainer.train()
```

---

## Python Client Example

```python
import requests

BASE = "http://localhost:7860"

# Start episode
r = requests.post(f"{BASE}/reset", json={"task_id": "task2", "seed": 99})
session_id = r.json()["session_id"]
obs = r.json()["observation"]

print(obs["disruption_description"])

# Agent loop
while not obs["done"]:
    # Your agent logic here
    action = {
        "session_id": session_id,
        "action_type": "expedite_shipment",
        "target_id": obs["affected_shipments"][0]["shipment_id"],
        "explanation": "Critical pharma cargo must meet 2-day deadline."
    }
    r = requests.post(f"{BASE}/step", json=action)
    obs = r.json()["observation"]
    
    if r.json()["done"]:
        result = r.json()["info"]["episode_result"]
        print(f"Final reward: {result['final_reward']:.3f}")
        print(f"Grader feedback: {result['grader_feedback']}")
```

---

## Baseline Results (seed=42)

| Task | Reward | Success | Cost (₹) | Turns |
|------|--------|---------|----------|-------|
| task1 | 0.653 | ✓ | 60,000 | 1 |
| task2 | 0.382 | ✗ | 78,400 | 3 |
| task3 | 0.183 | ✗ | 48,000 | 4 |

Reproduced by running: `python baseline.py`

---

## Project Structure

```
GatiAI/
├── models.py         # Typed Pydantic models (actions, observations, state)
├── scenarios.py      # Synthetic Indian supply chain scenario generator
├── graders.py        # Novel multi-dimensional graders (correctness + cost + risk)
├── environment.py    # Core env class: reset() / step() / state()
├── server.py         # FastAPI server (HTTP + WebSocket)
├── baseline.py       # Reproducible baseline inference script
├── openenv.yaml      # OpenEnv spec file
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Real-World Impact

This environment directly addresses India's logistics crisis:
- **₹380B** logistics market with **14–16% GDP** cost (vs 8–10% global benchmark)
- **15–18%** of agricultural produce lost to supply chain failures annually
- Zero open-source trained agents for real-time disruption response
- Directly applicable to India's ULIP platform (30 integrated logistics systems)

---

## License

Apache 2.0 — free to use, modify, and deploy.

All scenario data is **100% synthetic** — no real company information, no PII.
