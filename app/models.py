from pydantic import BaseModel
from typing import Any


class Observation(BaseModel):
    dataset_preview: list[dict]       # first 5 rows as list of dicts
    column_stats: dict                # per-column: dtype, null_count, duplicate_count, outlier_count
    issues_remaining: int             # total issues left unfixed
    step_count: int
    task_id: str
    task_description: str
    available_actions: list[str]


class Action(BaseModel):
    action_type: str   # fill_missing, drop_duplicates, fix_dtype, remove_outliers, normalize_format, done
    column: str | None = None
    params: dict = {}


class Reward(BaseModel):
    value: float           # 0.0 to 1.0
    reason: str
    quality_delta: float   # improvement from last step
    done: bool


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict
