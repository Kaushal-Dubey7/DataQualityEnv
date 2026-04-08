---
title: Data Quality Env
emoji: 🧹
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
---

# DataQualityEnv 🧹

> **A production-ready OpenEnv RL environment for training AI agents to audit and fix messy datasets.**

---

## 1. Environment Description & Motivation

Real-world data pipelines are routinely polluted with missing values, duplicate records, wrong data types, statistical outliers, and inconsistent formats. Data teams spend a disproportionate amount of time on these repetitive auditing tasks.

**DataQualityEnv** turns data cleaning into a structured RL problem:
- An agent receives a partially corrupted synthetic dataset.
- It inspects the dataset via a rich `Observation` including per-column statistics and a dataset preview.
- It applies structured `Actions` (fill missing, drop duplicates, fix dtypes, etc.) one at a time.
- It receives a reward signal proportional to the overall data quality score it achieves.
- The episode ends when the dataset reaches a passing quality threshold, the agent signals `done`, or it exhausts its step budget.

This framing is directly applicable to AutoML pipelines, data lake governance tools, and ETL automation.

---

## 2. Action Space

| `action_type` | Description | Key `params` / `column` |
|---|---|---|
| `fill_missing` | Fill null values with column mean (numeric) or mode (categorical) | `column` (optional, applies to all cols if omitted) |
| `drop_duplicates` | Remove all duplicate rows from the dataset | — |
| `fix_dtype` | Attempt to cast a column to its correct numeric or datetime type | `column` (optional) |
| `remove_outliers` | Clip values beyond ±3 standard deviations to the clipping bounds | `column` (optional) |
| `normalize_format` | Standardize date strings to `YYYY-MM-DD` | `column` (optional) |
| `done` | Signal that the agent believes the dataset is clean; ends the episode | — |

All actions accept an optional `column` field and an arbitrary `params` dict for future extensibility.

---

## 3. Observation Space

| Field | Type | Description |
|---|---|---|
| `dataset_preview` | `list[dict]` | First 5 rows of the current dataset as a list of row dicts |
| `column_stats` | `dict` | Per-column stats: `dtype`, `null_count`, `duplicate_count`, `outlier_count` |
| `issues_remaining` | `int` | Estimated total number of data quality issues still present |
| `step_count` | `int` | Number of steps taken so far in this episode |
| `task_id` | `str` | Identifier of the active task |
| `task_description` | `str` | Human-readable task objective |
| `available_actions` | `list[str]` | List of valid action types the agent may use |

---

## 4. Task Descriptions

| Task ID | Difficulty | Objective | Max Steps | Passing Score |
|---|---|---|---|---|
| `null_hunter` | Easy | Fix all missing values in the employee dataset | 15 | > 0.90 |
| `full_cleanup` | Medium | Remove duplicates, fix missing values, and correct data types | 25 | > 0.85 |
| `master_audit` | Hard | Fix nulls, duplicates, dtypes, outliers, and standardize all date formats to YYYY-MM-DD | 40 | > 0.88 |

**Datasets:**
- `null_hunter` — 100 rows × 5 columns, 10 % nulls
- `full_cleanup` — 200 rows × 7 columns, nulls + 15 % duplicates + 2 wrong-dtype columns
- `master_audit` — 300 rows × 10 columns, all of the above + 3-sigma outliers + mixed date formats

---

## 5. Reward Function

```
reward = quality_score(df, task_id)            # absolute quality, 0.0–1.0
       − 0.05  if quality_delta ≤ 0            # penalise no-progress steps
       − 0.02  if action is a no-op            # penalise wasted actions
```

The **quality score** is a weighted average across four dimensions:

| Dimension | Weight | Formula |
|---|---|---|
| Null score | 0.30 | `1 − (null_cells / total_cells)` |
| Duplicate score | 0.25 | `1 − (duplicate_rows / total_rows)` |
| Dtype score | 0.25 | `correct_dtype_columns / total_columns` |
| Outlier score | 0.20 | `1 − min(outlier_cells / total_cells, 1.0)` |

The **task graders** apply task-specific weightings on top of this global score, adding a date-format compliance term for `master_audit`.

Reward is always clamped to `[0.0, 1.0]`.

---

## 6. Setup & Usage

### Prerequisites
- Docker ≥ 20 **or** Python 3.11+

### Docker (recommended)

```bash
# Build
docker build -t data-quality-env .

# Run server (port 7860)
docker run -p 7860:7860 data-quality-env
```

### Local (without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### API Examples

**Health check**
```bash
curl http://localhost:7860/health
# {"status":"ok"}
```

**List tasks**
```bash
curl http://localhost:7860/tasks
```

**Reset environment**
```bash
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "null_hunter"}'
```

**Take a step**
```bash
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"action_type": "fill_missing", "column": null, "params": {}}'
```

**Inspect current state**
```bash
curl http://localhost:7860/state
```

### Running inference.py

Set the required environment variables, then:

```bash
export API_BASE_URL="http://localhost:7860"   # DataQualityEnv server
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="hf_..."

python inference.py
```

---

## 7. Baseline Scores

These scores represent a rule-based greedy agent that always applies the most impactful action first.

| Task | Baseline Score | Passed |
|---|---|---|
| `null_hunter` | 0.91 | ✅ |
| `full_cleanup` | 0.86 | ✅ |
| `master_audit` | 0.82 | ❌ |

---

## 8. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | ✅ | Full base URL of the DataQualityEnv server (e.g. `http://localhost:7860`) — also used as the OpenAI-compatible endpoint for the LLM |
| `MODEL_NAME` | ✅ | HuggingFace Inference API model name (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) |
| `HF_TOKEN` | ✅ | HuggingFace API token for authenticating LLM requests |

> ⚠️ **Never hard-code credentials.** All secrets must be injected via environment variables.

---

## Project Structure

```
DataQualityEnv/
├── app/
│   ├── __init__.py
│   ├── main.py           ← FastAPI app (port 7860)
│   ├── models.py         ← Pydantic v2 models
│   ├── environment.py    ← Core RL environment logic
│   ├── tasks.py          ← Task definitions and graders
│   └── datasets.py       ← Synthetic dataset generator
├── inference.py          ← LLM agent runner (root level)
├── openenv.yaml          ← OpenEnv manifest
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## License

MIT
