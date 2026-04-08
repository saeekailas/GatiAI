import os
import json
import requests
from openai import OpenAI

# Required Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")  # If needed for HF calls

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=API_BASE_URL)


def choose_action(observation: dict) -> str:
    prompt = f"""
You are an AI Logistics Manager for GatiAI.
Current Environment State: {json.dumps(observation, indent=2)}

Available actions:
- select_supplier
- reroute_shipment
- expedite_shipment
- delay_order
- request_info
- escalate_to_human
- split_order

Choose the best action_type from the list above.
Return ONLY the action_type as a single string.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def run_inference():
    print("[START]")
    
    base_url = "http://localhost:7860"
    tasks = ["task1", "task2", "task3"]
    
    for task_id in tasks:
        reset_resp = requests.post(f"{base_url}/reset", json={"task_id": task_id})
        if reset_resp.status_code != 200:
            raise RuntimeError(f"Reset failed for {task_id}: {reset_resp.text}")
        
        data = reset_resp.json()
        session_id = data["session_id"]
        obs = data["observation"]
        done = False
        step_count = 0
        total_reward = 0.0
        
        while not done and step_count < 20:  # Safety limit
            action_type = choose_action(obs)
            if not action_type:
                raise RuntimeError(f"OpenAI returned empty action_type for {task_id}")
            
            action_req = {
                "session_id": session_id,
                "action_type": action_type,
                "target_id": None,
                "parameters": {},
                "explanation": f"LLM chose {action_type}"
            }
            
            if action_type == "select_supplier" and obs.get("available_suppliers"):
                action_req["target_id"] = obs["available_suppliers"][0]["supplier_id"]
                action_req["parameters"] = {"quantity": 500}
            elif action_type == "reroute_shipment" and obs.get("affected_shipments"):
                action_req["target_id"] = obs["affected_shipments"][0]["shipment_id"]
            elif action_type == "expedite_shipment" and obs.get("affected_shipments"):
                action_req["target_id"] = obs["affected_shipments"][0]["shipment_id"]
            elif action_type == "delay_order" and obs.get("affected_shipments"):
                action_req["target_id"] = obs["affected_shipments"][0]["shipment_id"]
                action_req["parameters"] = {"delay_days": 7}
            elif action_type == "request_info":
                action_req["parameters"] = {"query_key": "stock_query_steel"}
            
            step_resp = requests.post(f"{base_url}/step", json=action_req)
            if step_resp.status_code != 200:
                raise RuntimeError(f"Step failed for {task_id}: {step_resp.text}")
            
            step_data = step_resp.json()
            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]
            total_reward += reward
            
            print(f"[STEP] {json.dumps({'step': step_count + 1, 'reward': reward, 'action': action_type})}")
            step_count += 1
        
    print("[END]")


if __name__ == "__main__":
    run_inference()