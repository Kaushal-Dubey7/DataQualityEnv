import os
import json
import datetime
import time

from openai import OpenAI

# ── Credentials ───────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-api-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

import requests

BASE = API_BASE_URL.rstrip("/")   # DataQualityEnv server URL, e.g. http://localhost:7860

client = OpenAI(
    api_key=API_BASE_URL,          # HF_TOKEN forwarded via OpenAI client
    base_url=API_BASE_URL,
)

# Override OpenAI client to use HF_TOKEN
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

SYSTEM_PROMPT = (
    "You are a data quality agent. You receive a dataset observation and must "
    "choose the best action to fix data quality issues. Always respond with "
    "ONLY a valid JSON matching: "
    "{\"action_type\": str, \"column\": str|null, \"params\": dict} "
    "Available action_types: fill_missing, drop_duplicates, fix_dtype, "
    "remove_outliers, normalize_format, done "
    "Choose 'done' only when quality looks good or you're stuck."
)

TASK_META = {
    "null_hunter":  "easy",
    "full_cleanup": "medium",
    "master_audit": "hard",
}

DEFAULT_ACTION = {"action_type": "fill_missing", "column": None, "params": {}}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _reset(task_id: str) -> dict:
    resp = requests.post(f"{BASE}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _step(action: dict) -> dict:
    resp = requests.post(f"{BASE}/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _llm_action(observation: dict) -> dict:
    """Ask the LLM for the next action given the current observation."""
    user_msg = json.dumps(observation, default=str)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=256,
            temperature=0.2,
        )
        raw = completion.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        action = json.loads(raw)
        # Validate required key
        if "action_type" not in action:
            raise ValueError("Missing action_type")
        return action

    except Exception as exc:
        print(f"  [WARN] LLM returned invalid response ({exc}), using default action.", flush=True)
        return DEFAULT_ACTION


# ──────────────────────────────────────────────────────────────────────────────
# Single-task runner
# ──────────────────────────────────────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    difficulty = TASK_META.get(task_id, "unknown")

    print(
        f'[START] {json.dumps({"task_id": task_id, "difficulty": difficulty, "timestamp": _now()})}',
        flush=True,
    )

    # Reset environment
    try:
        observation = _reset(task_id)
    except Exception as e:
        print(f"  [ERROR] _reset request failed: {e}", flush=True)
        return {"task_id": task_id, "final_score": 0.0, "steps_taken": 0, "passed": False}
    steps_taken = 0
    final_score = 0.0
    done = False

    while not done:
        action = _llm_action(observation)

        try:
            step_resp = _step(action)
        except Exception as exc:
            print(f"  [WARN] /step request failed: {exc}. Using default action.", flush=True)
            action = DEFAULT_ACTION
            try:
                step_resp = _step(action)
            except Exception:
                break

        reward_obj  = step_resp["reward"]
        obs_obj     = step_resp["observation"]
        done        = step_resp["done"]
        steps_taken += 1

        reward_val     = reward_obj["value"]
        quality_delta  = reward_obj["quality_delta"]
        issues_rem     = obs_obj["issues_remaining"]
        final_score    = step_resp.get("info", {}).get("raw_quality", reward_val)

        print(
            f'[STEP] {json.dumps({"step": steps_taken, "action": action, "reward": round(reward_val, 6), "quality_delta": round(quality_delta, 6), "issues_remaining": issues_rem, "done": done})}',
            flush=True,
        )

        observation = obs_obj

        if done:
            break

    # Determine task pass status from the environment's passing score thresholds
    passing_scores = {"null_hunter": 0.90, "full_cleanup": 0.85, "master_audit": 0.88}
    passed = final_score >= passing_scores.get(task_id, 0.85)

    print(
        f'[END] {json.dumps({"task_id": task_id, "final_score": round(final_score, 6), "steps_taken": steps_taken, "passed": passed, "timestamp": _now()})}',
        flush=True,
    )

    return {"task_id": task_id, "final_score": final_score, "steps_taken": steps_taken, "passed": passed}


# ──────────────────────────────────────────────────────────────────────────────
# Main – run all 3 tasks sequentially
# ──────────────────────────────────────────────────────────────────────────────

def main():
    task_ids = ["null_hunter", "full_cleanup", "master_audit"]
    results  = []
    total_start = time.time()

    for task_id in task_ids:
        result = run_task(task_id)
        results.append(result)

        elapsed = time.time() - total_start
        if elapsed > 18 * 60:   # 18-minute safety cutoff (< 20 min limit)
            print("[WARN] Approaching 20-minute limit – stopping early.", flush=True)
            break

    tasks_passed  = sum(1 for r in results if r["passed"])
    average_score = sum(r["final_score"] for r in results) / len(results) if results else 0.0
    total_steps   = sum(r["steps_taken"] for r in results)

    print(
        f'[SUMMARY] {json.dumps({"tasks_passed": tasks_passed, "average_score": round(average_score, 6), "total_steps": total_steps})}',
        flush=True,
    )


if __name__ == "__main__":
    main()
