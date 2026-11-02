"""
Dataset ingestion, validation, and statistical profiling module.
"""
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np


def load_csv(file_or_path) -> pd.DataFrame:
    """
    Safely load a CSV file into a pandas DataFrame.
    Supports file paths, string paths, and Streamlit UploadedFile objects.
    """
    try:
        df = pd.read_csv(file_or_path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_or_path, encoding="latin1")
        except Exception as e:
            raise ValueError(f"Could not decode CSV file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {str(e)}")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Clean column names (strip leading/trailing whitespace)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically classifies columns into:
    - categorical: object, category, or integer with few unique values (< 10)
    - numerical: int, float with many unique values
    - id_like: unique identifier candidate columns
    - datetime: parsed or parseable dates
    """
    categorical_cols = []
    numerical_cols = []
    id_cols = []
    datetime_cols = []

    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)

        # Check if ID-like: high uniqueness (> 95% unique) and contains 'id' or is string/hash
        is_id_named = any(k in col.lower() for k in ["id", "uuid", "guid", "key", "index", "code"])
        if is_id_named and n_unique >= 0.8 * n_rows and n_rows > 15:
            id_cols.append(col)
            continue

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue
        elif series.dtype == object and n_unique > 15:
            # Try to see if it's a date string
            sample = series.dropna().head(10)
            try:
                pd.to_datetime(sample, errors="raise")
                datetime_cols.append(col)
                continue
            except (ValueError, TypeError):
                pass

        # Check numerical vs categorical
        if pd.api.types.is_numeric_dtype(series):
            # If numeric but has very few unique values (e.g. 0/1 binary or status codes 1-4)
            if n_unique <= 6 and n_rows >= 30:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "categorical": categorical_cols,
        "numerical": numerical_cols,
        "id_like": id_cols,
        "datetime": datetime_cols
    }


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary metrics and column-level profiling information.
    """
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / n_rows) * 100, 2) if n_rows > 0 else 0.0

    memory_bytes = df.memory_usage(deep=True).sum()
    if memory_bytes > 1024 * 1024:
        memory_str = f"{memory_bytes / (1024 * 1024):.2f} MB"
    else:
        memory_str = f"{memory_bytes / 1024:.2f} KB"

    detected_types = detect_column_types(df)

    col_details = []
    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        n_unique = int(series.nunique(dropna=True))
        n_missing = int(series.isna().sum())
        missing_col_pct = round((n_missing / n_rows) * 100, 2) if n_rows > 0 else 0.0

        if col in detected_types["categorical"]:
            cat_type = "Categorical"
        elif col in detected_types["numerical"]:
            cat_type = "Numerical"
        elif col in detected_types["id_like"]:
            cat_type = "Identifier (ID)"
        elif col in detected_types["datetime"]:
            cat_type = "Datetime"
        else:
            cat_type = "Other"

        # Representative sample preview
        sample_vals = series.dropna().unique()[:3]
        sample_str = ", ".join(str(v) for v in sample_vals)
        if len(sample_str) > 35:
            sample_str = sample_str[:32] + "..."

        col_details.append({
            "Column Name": col,
            "Detected Role": cat_type,
            "Pandas Dtype": dtype_str,
            "Unique Values": n_unique,
            "Missing Count": n_missing,
            "Missing (%)": missing_col_pct,
            "Sample Values": sample_str
        })

    col_df = pd.DataFrame(col_details)

    return {
        "rows": n_rows,
        "columns": n_cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "memory_usage": memory_str,
        "column_classification": detected_types,
        "column_profile_df": col_df
    }
