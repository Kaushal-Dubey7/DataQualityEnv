import re
import numpy as np
import pandas as pd
from scipy import stats

from app.models import Action, Observation, Reward, StepResponse
from app.datasets import SyntheticDatasetGenerator
from app.tasks import TaskManager, TASK_REGISTRY

_generator = SyntheticDatasetGenerator()
_task_mgr = TaskManager()

# Regex for YYYY-MM-DD
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

AVAILABLE_ACTIONS = [
    "fill_missing",
    "drop_duplicates",
    "fix_dtype",
    "remove_outliers",
    "normalize_format",
    "done",
]


def _parse_any_date(val: str) -> str | None:
    """Try to parse a date string in multiple formats and return YYYY-MM-DD."""
    for fmt in ("%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(val, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    try:
        return pd.to_datetime(val, infer_datetime_format=True).strftime("%Y-%m-%d")
    except Exception:
        return None


class DataQualityEnvironment:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.original_df: pd.DataFrame = None
        self.task_id: str = None
        self.step_count: int = 0
        self.done: bool = False
        self.prev_quality: float = 0.0

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self, task_id: str = "null_hunter") -> Observation:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id: {task_id!r}")

        self.task_id = task_id
        self.step_count = 0
        self.done = False

        self.df = _generator.generate(task_id, seed=42)
        self.original_df = self.df.copy()
        self.prev_quality = _generator.compute_quality_score(self.df, task_id)

        return self._build_observation()

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: Action) -> StepResponse:
        if self.df is None or self.task_id is None:
            raise RuntimeError("Environment not initialised. Call /reset first.")

        if self.done:
            obs = self._build_observation()
            reward = Reward(
                value=self.prev_quality,
                reason="Episode already finished.",
                quality_delta=0.0,
                done=True,
            )
            return StepResponse(observation=obs, reward=reward, done=True, info={})

        self.step_count += 1
        penalty = 0.0
        reason = ""
        is_noop = False

        action_type = action.action_type.strip().lower()

        # ---- Apply action ----
        if action_type == "fill_missing":
            changed = self._action_fill_missing(action)
            if not changed:
                is_noop = True
                reason = "No missing values to fill."
            else:
                reason = f"Filled missing values in column(s)."

        elif action_type == "drop_duplicates":
            before = len(self.df)
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            after = len(self.df)
            if before == after:
                is_noop = True
                reason = "No duplicates found."
            else:
                reason = f"Dropped {before - after} duplicate rows."

        elif action_type == "fix_dtype":
            changed = self._action_fix_dtype(action)
            if not changed:
                is_noop = True
                reason = "No dtype changes applied."
            else:
                reason = "Fixed column dtype(s)."

        elif action_type == "remove_outliers":
            changed = self._action_remove_outliers(action)
            if not changed:
                is_noop = True
                reason = "No outliers removed."
            else:
                reason = "Clipped outlier values."

        elif action_type == "normalize_format":
            changed = self._action_normalize_format(action)
            if not changed:
                is_noop = True
                reason = "No format normalisation needed."
            else:
                reason = "Normalised date formats to YYYY-MM-DD."

        elif action_type == "done":
            reason = "Agent signalled completion."
            self.done = True

        else:
            is_noop = True
            reason = f"Unknown action_type: {action_type!r}."

        # ---- Score ----
        new_quality = _generator.compute_quality_score(self.df, self.task_id)
        quality_delta = new_quality - self.prev_quality

        reward_value = float(new_quality)

        if quality_delta <= 0.0 and not self.done:
            penalty += 0.05
            reason += " [Penalty: no improvement]"

        if is_noop:
            penalty += 0.02
            reason += " [Penalty: no-op]"

        reward_value = float(np.clip(reward_value - penalty, 0.0, 1.0))

        # ---- Check termination ----
        task_meta = TASK_REGISTRY[self.task_id]
        if (
            self.step_count >= task_meta["max_steps"]
            or action_type == "done"
            or new_quality >= task_meta["passing_score"]
        ):
            self.done = True

        self.prev_quality = new_quality

        obs = self._build_observation()
        reward = Reward(
            value=reward_value,
            reason=reason,
            quality_delta=quality_delta,
            done=self.done,
        )
        return StepResponse(
            observation=obs,
            reward=reward,
            done=self.done,
            info={
                "raw_quality": new_quality,
                "step_count": self.step_count,
                "max_steps": task_meta["max_steps"],
                "passing_score": task_meta["passing_score"],
            },
        )

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        if self.df is None:
            return {"status": "not_initialised"}

        obs = self._build_observation()
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "done": self.done,
            "quality_score": self.prev_quality,
            "observation": obs.model_dump(),
        }

    # ------------------------------------------------------------------ #
    # Private helpers – actions
    # ------------------------------------------------------------------ #
    def _action_fill_missing(self, action: Action) -> bool:
        cols = [action.column] if action.column and action.column in self.df.columns else self.df.columns.tolist()
        changed = False
        for col in cols:
            if self.df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_val = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(fill_val)
                else:
                    fill_val = self.df[col].mode()
                    fill_val = fill_val.iloc[0] if not fill_val.empty else "unknown"
                    self.df[col] = self.df[col].fillna(fill_val)
                changed = True
        return changed

    def _action_fix_dtype(self, action: Action) -> bool:
        cols = [action.column] if action.column and action.column in self.df.columns else self.df.columns.tolist()
        changed = False
        for col in cols:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                converted = pd.to_numeric(self.df[col], errors="coerce")
                if converted.notna().sum() > self.df[col].notna().sum() * 0.5:
                    self.df[col] = converted
                    changed = True
            if not changed and not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                converted = pd.to_datetime(self.df[col], errors="coerce", infer_datetime_format=True)
                if converted.notna().sum() > self.df[col].notna().sum() * 0.5:
                    # Keep as string YYYY-MM-DD after conversion
                    self.df[col] = converted.dt.strftime("%Y-%m-%d")
                    changed = True
        return changed

    def _action_remove_outliers(self, action: Action) -> bool:
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if action.column and action.column in num_cols:
            num_cols = [action.column]

        if not num_cols:
            return False

        changed = False
        for col in num_cols:
            col_data = self.df[col].dropna()
            if len(col_data) < 2:
                continue
            mean, std = col_data.mean(), col_data.std()
            if std == 0:
                continue
            lower, upper = mean - 3 * std, mean + 3 * std
            before_sum = self.df[col].sum()
            self.df[col] = self.df[col].clip(lower=lower, upper=upper)
            if self.df[col].sum() != before_sum:
                changed = True
        return changed

    def _action_normalize_format(self, action: Action) -> bool:
        # Operate on the column named in action, or try all object columns
        candidate_cols = (
            [action.column]
            if action.column and action.column in self.df.columns
            else self.df.select_dtypes(include=["object"]).columns.tolist()
        )
        changed = False
        for col in candidate_cols:
            # Check if this column looks like dates
            sample = self.df[col].dropna().astype(str).head(20)
            if not self._looks_like_dates(sample):
                continue
            new_vals = self.df[col].apply(
                lambda v: _parse_any_date(str(v)) if pd.notna(v) else v
            )
            if not new_vals.equals(self.df[col]):
                self.df[col] = new_vals
                changed = True
        return changed

    @staticmethod
    def _looks_like_dates(sample: pd.Series) -> bool:
        date_like = re.compile(
            r"\d{2,4}[-/]\d{2}[-/]\d{2,4}"
        )
        hits = sample.apply(lambda v: bool(date_like.search(str(v)))).sum()
        return hits >= len(sample) * 0.5

    # ------------------------------------------------------------------ #
    # Private helpers – observation builder
    # ------------------------------------------------------------------ #
    def _build_observation(self) -> Observation:
        preview = self.df.head(5).replace({np.nan: None}).to_dict(orient="records")

        column_stats: dict = {}
        num_df = self.df.select_dtypes(include=[np.number])
        for col in self.df.columns:
            col_data = self.df[col]
            null_count = int(col_data.isnull().sum())
            dup_count = int(col_data.duplicated().sum())

            outlier_count = 0
            if col in num_df.columns and col_data.dropna().shape[0] > 1:
                col_clean = col_data.dropna()
                z = np.abs(stats.zscore(col_clean))
                outlier_count = int((z > 3).sum())

            column_stats[col] = {
                "dtype": str(col_data.dtype),
                "null_count": null_count,
                "duplicate_count": dup_count,
                "outlier_count": outlier_count,
            }

        issues_remaining = _generator.count_issues(self.df, self.task_id)
        task_meta = TASK_REGISTRY[self.task_id]

        return Observation(
            dataset_preview=preview,
            column_stats=column_stats,
            issues_remaining=issues_remaining,
            step_count=self.step_count,
            task_id=self.task_id,
            task_description=task_meta["description"],
            available_actions=AVAILABLE_ACTIONS,
        )
