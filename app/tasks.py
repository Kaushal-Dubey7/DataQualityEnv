import re
import numpy as np
import pandas as pd
from scipy import stats


TASK_REGISTRY: dict[str, dict] = {
    "null_hunter": {
        "id": "null_hunter",
        "difficulty": "easy",
        "description": "Fix all missing values in the employee dataset",
        "max_steps": 15,
        "passing_score": 0.90,
    },
    "full_cleanup": {
        "id": "full_cleanup",
        "difficulty": "medium",
        "description": "Remove duplicates, fix missing values, and correct data types",
        "max_steps": 25,
        "passing_score": 0.85,
    },
    "master_audit": {
        "id": "master_audit",
        "difficulty": "hard",
        "description": (
            "Perform a complete audit: fix nulls, duplicates, dtypes, "
            "outliers, and standardize all date formats to YYYY-MM-DD"
        ),
        "max_steps": 40,
        "passing_score": 0.88,
    },
}

# Regex for YYYY-MM-DD
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class TaskManager:
    def get_task(self, task_id: str) -> dict:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id: {task_id!r}")
        return TASK_REGISTRY[task_id]

    def list_tasks(self) -> list[dict]:
        return list(TASK_REGISTRY.values())

    # ------------------------------------------------------------------ #
    # Graders
    # ------------------------------------------------------------------ #
    def grade(self, df: pd.DataFrame, task_id: str) -> float:
        """Return a deterministic score 0.0–1.0 for the given df and task."""
        if task_id == "null_hunter":
            return self._grade_null_hunter(df)
        elif task_id == "full_cleanup":
            return self._grade_full_cleanup(df)
        elif task_id == "master_audit":
            return self._grade_master_audit(df)
        raise ValueError(f"Unknown task_id: {task_id!r}")

    # --- Task 1 grader ---
    def _grade_null_hunter(self, df: pd.DataFrame) -> float:
        rows, cols = df.shape
        if rows == 0:
            return 0.0

        null_cells = int(df.isnull().sum().sum())
        null_score = 1.0 - (null_cells / (rows * cols))

        # Small bonus for row count preservation
        row_penalty = max(0.0, 1.0 - abs(rows - 100) / 100)

        score = 0.85 * null_score + 0.15 * row_penalty
        return round(float(np.clip(score, 0.0, 1.0)), 6)

    # --- Task 2 grader ---
    def _grade_full_cleanup(self, df: pd.DataFrame) -> float:
        rows, cols = df.shape
        if rows == 0:
            return 0.0

        null_cells = int(df.isnull().sum().sum())
        null_score = 1.0 - (null_cells / (rows * cols))

        dup_rows = int(df.duplicated().sum())
        dup_score = 1.0 - (dup_rows / rows)

        numeric_targets = ["age", "performance"]
        correct_dtypes = sum(
            1 for c in numeric_targets
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        )
        dtype_score = correct_dtypes / len(numeric_targets)

        score = 0.35 * null_score + 0.35 * dup_score + 0.30 * dtype_score
        return round(float(np.clip(score, 0.0, 1.0)), 6)

    # --- Task 3 grader ---
    def _grade_master_audit(self, df: pd.DataFrame) -> float:
        rows, cols = df.shape
        if rows == 0:
            return 0.0

        # Null component
        null_cells = int(df.isnull().sum().sum())
        null_score = 1.0 - (null_cells / (rows * cols))

        # Duplicate component
        dup_rows = int(df.duplicated().sum())
        dup_score = 1.0 - (dup_rows / rows)

        # Dtype component
        numeric_targets = ["age", "test_score"]
        correct_dtypes = sum(
            1 for c in numeric_targets
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        )
        dtype_score = correct_dtypes / len(numeric_targets)

        # Outlier component
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            outlier_score = 1.0
        else:
            z = np.abs(stats.zscore(num_df.dropna()))
            outlier_cells = int((z > 3).sum())
            outlier_score = 1.0 - min(outlier_cells / (rows * cols), 1.0)

        # Date format component
        if "start_date" in df.columns:
            date_col = df["start_date"].dropna().astype(str)
            if len(date_col) > 0:
                matching = date_col.apply(lambda v: bool(_DATE_PATTERN.match(v))).sum()
                date_score = matching / len(date_col)
            else:
                date_score = 1.0
        else:
            date_score = 1.0

        score = (
            0.25 * null_score
            + 0.20 * dup_score
            + 0.20 * dtype_score
            + 0.20 * outlier_score
            + 0.15 * date_score
        )
        return round(float(np.clip(score, 0.0, 1.0)), 6)

    def is_passing(self, score: float, task_id: str) -> bool:
        return score >= TASK_REGISTRY[task_id]["passing_score"]
