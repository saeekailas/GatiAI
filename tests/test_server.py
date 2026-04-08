from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "active_sessions" in body


def test_tasks_and_schema_endpoints():
    tasks = client.get("/tasks")
    assert tasks.status_code == 200
    assert "tasks" in tasks.json()

    schema = client.get("/schema")
    assert schema.status_code == 200
    data = schema.json()
    assert "action" in data
    assert "observation" in data
    assert "state" in data


def test_reset_and_step_cycle():
    reset_resp = client.post("/reset", json={"task_id": "task1", "seed": 42})
    assert reset_resp.status_code == 200
    payload = reset_resp.json()
    assert payload["session_id"]
    assert payload["observation"]["task_id"] == "task1"

    session_id = payload["session_id"]
    action = {
        "session_id": session_id,
        "action_type": "select_supplier",
        "target_id": payload["observation"]["available_suppliers"][0]["supplier_id"],
        "parameters": {"quantity": 500},
        "explanation": "Testing step action with a valid supplier.",
    }
    step_resp = client.post("/step", json=action)
    assert step_resp.status_code == 200
    step_data = step_resp.json()
    assert "reward" in step_data
    assert "observation" in step_data
    assert "done" in step_data


def test_state_route_after_reset():
    reset_resp = client.post("/reset", json={"task_id": "task1", "seed": 123})
    session_id = reset_resp.json()["session_id"]
    state_resp = client.get(f"/state/{session_id}")
    assert state_resp.status_code == 200
    state = state_resp.json()
    assert state["task_id"] == "task1"
    assert state["turn"] == 0
