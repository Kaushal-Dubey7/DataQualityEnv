import numpy as np
import pandas as pd
from scipy import stats


class SyntheticDatasetGenerator:
    """Generates deterministic synthetic messy datasets for each task."""

    def generate(self, task_id: str, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)

        if task_id == "null_hunter":
            return self._generate_task1(rng)
        elif task_id == "full_cleanup":
            return self._generate_task2(rng)
        elif task_id == "master_audit":
            return self._generate_task3(rng)
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

    # ------------------------------------------------------------------ #
    # Task 1 – 100 rows, 5 columns, 10 % nulls only
    # ------------------------------------------------------------------ #
    def _generate_task1(self, rng: np.random.Generator) -> pd.DataFrame:
        n = 100
        df = pd.DataFrame({
            "employee_id":  rng.integers(1000, 9999, size=n).astype(float),
            "age":          rng.integers(22, 65,   size=n).astype(float),
            "salary":       rng.uniform(30_000, 120_000, size=n).round(2),
            "department":   rng.choice(["HR", "Eng", "Sales", "Finance"], size=n).tolist(),
            "years_exp":    rng.integers(0, 30, size=n).astype(float),
        })

        # Inject ~10 % nulls randomly across all cells
        total_cells = n * len(df.columns)
        null_count = int(total_cells * 0.10)
        flat_indices = rng.choice(total_cells, size=null_count, replace=False)
        for idx in flat_indices:
            row, col = divmod(int(idx), len(df.columns))
            df.iat[row, col] = np.nan

        return df

    # ------------------------------------------------------------------ #
    # Task 2 – 200 rows, 7 cols, nulls + 15 % dupes + 2 wrong-dtype cols
    # ------------------------------------------------------------------ #
    def _generate_task2(self, rng: np.random.Generator) -> pd.DataFrame:
        n_base = 170  # will be padded with duplicates to reach ~200
        df = pd.DataFrame({
            "employee_id": rng.integers(1000, 9999, size=n_base).astype(float),
            "age":         rng.integers(22, 65,   size=n_base).astype(float),
            "salary":      rng.uniform(30_000, 120_000, size=n_base).round(2),
            "department":  rng.choice(["HR", "Eng", "Sales", "Finance"], size=n_base).tolist(),
            "years_exp":   rng.integers(0, 30, size=n_base).astype(float),
            "start_date":  pd.date_range("2010-01-01", periods=n_base, freq="W").strftime("%Y-%m-%d").tolist(),
            "performance": rng.uniform(1, 5, size=n_base).round(2),
        })

        # Inject ~10 % nulls
        total_cells = n_base * len(df.columns)
        null_count = int(total_cells * 0.10)
        flat_idx = rng.choice(total_cells, size=null_count, replace=False)
        for idx in flat_idx:
            row, col = divmod(int(idx), len(df.columns))
            df.iat[row, col] = np.nan

        # Add ~15 % duplicates (duplicate existing rows)
        n_dupes = int(200 * 0.15)
        dup_rows = df.sample(n=n_dupes, random_state=42)
        df = pd.concat([df, dup_rows], ignore_index=True)

        # Wrong dtype columns: store numeric cols as strings
        df["age"] = df["age"].astype(str)          # should be numeric
        df["performance"] = df["performance"].astype(str)  # should be numeric

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # Task 3 – 300 rows, 10 cols, everything + outliers + inconsistent dates
    # ------------------------------------------------------------------ #
    def _generate_task3(self, rng: np.random.Generator) -> pd.DataFrame:
        n_base = 255

        ages     = rng.integers(22, 65,       size=n_base).astype(float)
        salaries = rng.uniform(30_000, 120_000, size=n_base).round(2)
        scores   = rng.uniform(0, 100,         size=n_base).round(2)
        hours    = rng.uniform(20, 60,         size=n_base).round(1)
        ratings  = rng.uniform(1, 5,           size=n_base).round(2)

        # Inject outliers (beyond 3-sigma)
        outlier_idx = rng.choice(n_base, size=8, replace=False)
        ages[outlier_idx[:2]]     = rng.uniform(200, 300, size=2)    # impossible ages
        salaries[outlier_idx[2:4]] = rng.uniform(1_000_000, 5_000_000, size=2)
        scores[outlier_idx[4:6]]   = rng.uniform(500, 1000, size=2)
        hours[outlier_idx[6:]]     = rng.uniform(300, 500, size=2)

        # Build two inconsistent date formats for the same column
        dates_fmt1 = pd.date_range("2010-01-01", periods=n_base // 2, freq="W").strftime("%d-%m-%Y").tolist()
        dates_fmt2 = pd.date_range("2014-01-01", periods=n_base - n_base // 2, freq="W").strftime("%Y/%m/%d").tolist()
        mixed_dates = dates_fmt1 + dates_fmt2
        rng.shuffle(mixed_dates)

        df = pd.DataFrame({
            "employee_id":  rng.integers(1000, 9999, size=n_base).astype(float),
            "age":          ages,
            "salary":       salaries,
            "department":   rng.choice(["HR", "Eng", "Sales", "Finance", "Legal"], size=n_base).tolist(),
            "years_exp":    rng.integers(0, 30, size=n_base).astype(float),
            "start_date":   mixed_dates,
            "performance":  ratings,
            "test_score":   scores,
            "hours_worked": hours,
            "region":       rng.choice(["North", "South", "East", "West"], size=n_base).tolist(),
        })

        # Inject ~10 % nulls
        total_cells = n_base * len(df.columns)
        null_count = int(total_cells * 0.10)
        flat_idx = rng.choice(total_cells, size=null_count, replace=False)
        for idx in flat_idx:
            row, col = divmod(int(idx), len(df.columns))
            df.iat[row, col] = np.nan

        # Add ~15 % duplicates
        n_dupes = int(300 * 0.15)
        dup_rows = df.sample(n=n_dupes, random_state=42)
        df = pd.concat([df, dup_rows], ignore_index=True)

        # Two wrong-dtype columns
        df["age"]        = df["age"].astype(str)
        df["test_score"] = df["test_score"].astype(str)

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # Quality scorer
    # ------------------------------------------------------------------ #
    def compute_quality_score(self, df: pd.DataFrame, task_id: str) -> float:
        rows, cols = df.shape
        if rows == 0 or cols == 0:
            return 0.0

        # --- null score ---
        null_cells = int(df.isnull().sum().sum())
        null_score = 1.0 - (null_cells / (rows * cols))

        # --- duplicate score ---
        dup_rows = int(df.duplicated().sum())
        duplicate_score = 1.0 - (dup_rows / rows)

        # --- dtype score ---
        numeric_cols = self._expected_numeric_cols(task_id)
        if numeric_cols:
            correct = sum(
                1 for c in numeric_cols
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            )
            dtype_score = correct / len(numeric_cols)
        else:
            dtype_score = 1.0

        # --- outlier score ---
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            outlier_score = 1.0
        else:
            z_scores = np.abs(stats.zscore(num_df.dropna()))
            outlier_cells = int((z_scores > 3).sum())
            outlier_score = 1.0 - min(outlier_cells / (rows * cols), 1.0)

        # Weighted average
        score = (
            0.30 * null_score
            + 0.25 * duplicate_score
            + 0.25 * dtype_score
            + 0.20 * outlier_score
        )
        return round(float(np.clip(score, 0.0, 1.0)), 6)

    def _expected_numeric_cols(self, task_id: str) -> list[str]:
        if task_id == "null_hunter":
            return ["employee_id", "age", "salary", "years_exp"]
        elif task_id == "full_cleanup":
            return ["employee_id", "age", "salary", "years_exp", "performance"]
        elif task_id == "master_audit":
            return ["employee_id", "age", "salary", "years_exp",
                    "performance", "test_score", "hours_worked"]
        return []

    def count_issues(self, df: pd.DataFrame, task_id: str) -> int:
        """Count total number of data quality issues present."""
        issues = 0

        # Nulls
        issues += int(df.isnull().sum().sum())

        # Duplicates
        issues += int(df.duplicated().sum())

        # Wrong dtypes
        numeric_cols = self._expected_numeric_cols(task_id)
        for c in numeric_cols:
            if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
                issues += 1

        # Outliers in numeric cols
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            z = np.abs(stats.zscore(num_df.dropna()))
            issues += int((z > 3).sum())

        return issues
