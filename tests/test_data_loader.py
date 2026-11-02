"""
Unit tests for data loader and dataset profiling module.
"""
import pytest
import pandas as pd
from src.data_loader import load_csv, detect_column_types, profile_dataset


def test_load_csv_valid(tmp_path):
    df_sample = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85.5, 90.0, 78.5]
    })
    csv_file = tmp_path / "test.csv"
    df_sample.to_csv(csv_file, index=False)

    df_loaded = load_csv(csv_file)
    assert df_loaded.shape == (3, 3)
    assert list(df_loaded.columns) == ["id", "name", "score"]


def test_detect_column_types():
    df = pd.DataFrame({
        "Employee_ID": [f"ID_{i}" for i in range(50)],
        "Age": [20 + i for i in range(50)],
        "Department": ["IT", "HR"] * 25,
        "Salary": [50000 + i * 100 for i in range(50)]
    })
    col_types = detect_column_types(df)
    assert "Employee_ID" in col_types["id_like"]
    assert "Department" in col_types["categorical"]
    assert "Age" in col_types["numerical"]
    assert "Salary" in col_types["numerical"]


def test_profile_dataset():
    df = pd.DataFrame({
        "a": [1, 2, None, 4],
        "b": ["x", "y", "x", "y"]
    })
    profile = profile_dataset(df)
    assert profile["rows"] == 4
    assert profile["columns"] == 2
    assert profile["missing_cells"] == 1
    assert profile["missing_pct"] == 12.5
    assert not profile["column_profile_df"].empty
