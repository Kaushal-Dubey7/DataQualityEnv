"""
Smoke test for DataQualityEnv API.
Run while the server is up:  python test_api.py
"""
import urllib.request
import json

BASE = "http://localhost:7860"


def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())


def post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def sep(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


# ── 1. Health ──────────────────────────────────────
sep("1. GET /health")
resp = get("/health")
print(f"  {resp}")
assert resp == {"status": "ok"}, "Health check failed!"

# ── 2. Tasks ──────────────────────────────────────
sep("2. GET /tasks")
tasks = get("/tasks")
for t in tasks:
    print(f"  {t['id']:15s}  difficulty={t['difficulty']:6s}  max_steps={t['max_steps']}  passing={t['passing_score']}")
assert len(tasks) == 3

# ── 3. Reset null_hunter ──────────────────────────
sep("3. POST /reset  (null_hunter)")
obs = post("/reset", {"task_id": "null_hunter"})
print(f"  task_id          : {obs['task_id']}")
print(f"  issues_remaining : {obs['issues_remaining']}")
print(f"  step_count       : {obs['step_count']}")
print(f"  available_actions: {obs['available_actions']}")
print(f"  dataset_preview[0]: {obs['dataset_preview'][0]}")
assert obs["task_id"] == "null_hunter"
assert obs["step_count"] == 0

# ── 4. Step — fill_missing ────────────────────────
sep("4. POST /step  (fill_missing)")
step = post("/step", {"action_type": "fill_missing", "column": None, "params": {}})
reward = step["reward"]
print(f"  reward.value     : {reward['value']:.6f}")
print(f"  quality_delta    : {reward['quality_delta']:+.6f}")
print(f"  issues_remaining : {step['observation']['issues_remaining']}")
print(f"  done             : {step['done']}")
print(f"  reason           : {reward['reason']}")
assert 0.0 <= reward["value"] <= 1.0, "Reward out of range!"

# ── 5. State ──────────────────────────────────────
sep("5. GET /state")
state = get("/state")
print(f"  quality_score : {state['quality_score']}")
print(f"  step_count    : {state['step_count']}")
print(f"  done          : {state['done']}")

# ── 6. Signal done ────────────────────────────────
sep("6. POST /step  (done)")
step = post("/step", {"action_type": "done", "column": None, "params": {}})
print(f"  reward.value  : {step['reward']['value']:.6f}")
print(f"  done          : {step['done']}")
assert step["done"] is True

# ── 7. full_cleanup walkthrough ───────────────────
sep("7. full_cleanup walkthrough")
obs = post("/reset", {"task_id": "full_cleanup"})
print(f"  issues at reset: {obs['issues_remaining']}")
actions = [
    {"action_type": "fill_missing",    "column": None,          "params": {}},
    {"action_type": "drop_duplicates", "column": None,          "params": {}},
    {"action_type": "fix_dtype",       "column": "age",         "params": {}},
    {"action_type": "fix_dtype",       "column": "performance", "params": {}},
]
for act in actions:
    s = post("/step", act)
    print(
        f"  {act['action_type']:20s}  "
        f"reward={s['reward']['value']:.4f}  "
        f"delta={s['reward']['quality_delta']:+.4f}  "
        f"issues={s['observation']['issues_remaining']}"
    )

# ── 8. master_audit reset check ───────────────────
sep("8. master_audit reset")
obs = post("/reset", {"task_id": "master_audit"})
print(f"  task_id          : {obs['task_id']}")
print(f"  issues_remaining : {obs['issues_remaining']}")
print(f"  cols : {list(obs['column_stats'].keys())}")

print("\n" + "="*50)
print("  ALL TESTS PASSED ✅")
print("="*50)
