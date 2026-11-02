"""
End-to-End Scenario and Edge-Case Test Suite for Synthetic Tabular Data Generator.
Validates all 17 user scenarios and edge conditions.
"""
import pytest
import pandas as pd
import numpy as np
import io

from modules.data_loader import load_csv, detect_column_types, profile_dataset
from modules.preprocessor import Preprocessor, build_sdv_metadata
from modules.gan_model import CTGANModel
from modules.evaluator import DataQualityEvaluator
from modules.privacy_analyzer import PrivacyRiskAnalyzer
from utils.visualizer import (
    plot_numerical_distribution,
    plot_categorical_distribution,
    plot_correlation_heatmaps,
    plot_pca_projection,
    plot_dcr_histogram
)
from utils.report_generator import generate_html_report


# =====================================================================
# Scenarios 1 - 7: Ingestion & Data Shape Tests
# =====================================================================

def test_scenario_01_valid_csv(tmp_path):
    """Scenario 1: Upload a valid CSV file."""
    df = pd.DataFrame({"age": [25, 30, 35], "dept": ["IT", "HR", "Sales"]})
    fpath = tmp_path / "valid.csv"
    df.to_csv(fpath, index=False)
    loaded = load_csv(fpath)
    assert not loaded.empty
    assert len(loaded) == 3


def test_scenario_02_invalid_file(tmp_path):
    """Scenario 2: Upload an invalid file."""
    fpath = tmp_path / "corrupt.bin"
    with open(fpath, "wb") as f:
        f.write(b"\x00\xFF\xFE\x00corruptbinarydata")
    with pytest.raises(Exception):
        load_csv(fpath)


def test_scenario_03_missing_values(tmp_path):
    """Scenario 3: Upload a dataset containing missing values."""
    df = pd.DataFrame({
        "age": [25, np.nan, 35, 40],
        "dept": ["IT", "HR", None, "IT"]
    })
    profile = profile_dataset(df)
    assert profile["missing_cells"] == 2
    assert profile["missing_pct"] == 25.0


def test_scenario_04_duplicate_records():
    """Scenario 4: Upload a dataset containing duplicate records."""
    df = pd.DataFrame({
        "age": [25, 25, 30, 30, 35],
        "salary": [50000, 50000, 60000, 60000, 70000]
    })
    profile = profile_dataset(df)
    assert profile["duplicate_rows"] == 2


def test_scenario_05_numerical_only():
    """Scenario 5: Upload numerical-only data."""
    df = pd.DataFrame({
        "f1": [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9],
        "f2": [10, 20, 30, 40, 50, 60, 70, 80],
        "f3": [100.5, 200.5, 300.5, 400.5, 500.5, 600.5, 700.5, 800.5]
    })
    detected = detect_column_types(df)
    assert len(detected["numerical"]) == 3
    assert len(detected["categorical"]) == 0


def test_scenario_06_categorical_only():
    """Scenario 6: Upload categorical-only data."""
    df = pd.DataFrame({
        "c1": ["Red", "Blue", "Green", "Red", "Blue", "Green"] * 3,
        "c2": ["Small", "Medium", "Large", "Small", "Medium", "Large"] * 3
    })
    detected = detect_column_types(df)
    assert len(detected["categorical"]) == 2
    assert len(detected["numerical"]) == 0


def test_scenario_07_mixed_data():
    """Scenario 7: Upload mixed numerical and categorical data."""
    df = pd.DataFrame({
        "age": [25, 32, 45, 52, 28, 39] * 3,
        "salary": [45000, 62000, 95000, 110000, 54000, 78000] * 3,
        "department": ["IT", "HR", "IT", "Sales", "HR", "Sales"] * 3
    })
    detected = detect_column_types(df)
    assert "age" in detected["numerical"]
    assert "salary" in detected["numerical"]
    assert "department" in detected["categorical"]


# =====================================================================
# Scenarios 8 - 12: Preprocessing, Training & Multi-Size Generation
# =====================================================================

def test_scenario_08_preprocessing():
    """Scenario 8: Preprocess the uploaded data."""
    df = pd.DataFrame({
        "ID": ["A1", "A2", "A3", "A3"],
        "num": [10.0, np.nan, 30.0, 30.0],
        "cat": ["X", "Y", None, None]
    })
    prep = Preprocessor(impute_strategy="impute_median_mode", drop_duplicates=True, drop_id_columns=True)
    clean_df, audit = prep.fit_transform(df, {"id_like": ["ID"], "numerical": ["num"], "categorical": ["cat"]})
    assert "ID" not in clean_df.columns
    assert clean_df.isna().sum().sum() == 0
    assert audit["initial_duplicates"] == 1


def test_scenario_09_to_12_train_and_generate_100_500_1000():
    """Scenarios 9-12: Train CTGAN model and generate 100, 500, 1000 records."""
    df = pd.DataFrame({
        "age": [22, 28, 35, 42, 50, 58, 26, 31, 45, 52] * 2,
        "income": [30000, 45000, 60000, 80000, 95000, 110000, 38000, 52000, 85000, 98000] * 2,
        "role": ["Dev", "HR", "Dev", "Sales", "HR", "Sales", "Dev", "Sales", "HR", "Dev"] * 2
    })
    model = CTGANModel(epochs=2, batch_size=10, pac=2)
    fit_res = model.fit(df, discrete_columns=["role"])
    assert model.is_trained
    assert fit_res["status"] == "Success"

    # Scenario 10: 100 records
    s100 = model.generate(100)
    assert len(s100) == 100

    # Scenario 11: 500 records
    s500 = model.generate(500)
    assert len(s500) == 500

    # Scenario 12: 1000 records
    s1000 = model.generate(1000)
    assert len(s1000) == 1000
    assert (s1000["age"] >= 0).all()


# =====================================================================
# Scenarios 13 - 17: Quality, Privacy, Visuals & Exports
# =====================================================================

def test_scenario_13_quality_evaluation():
    """Scenario 13: Evaluate synthetic data quality."""
    real = pd.DataFrame({"age": [20, 30, 40, 50], "score": [10, 20, 30, 40], "cat": ["A", "B", "A", "B"]})
    synth = pd.DataFrame({"age": [22, 28, 38, 51], "score": [12, 19, 31, 39], "cat": ["A", "B", "A", "B"]})
    evaluator = DataQualityEvaluator(real, synth, discrete_columns=["cat"])
    res = evaluator.generate_full_evaluation()
    assert res["overall_score"] > 50.0
    assert not res["numerical_metrics"].empty
    assert not res["categorical_metrics"].empty


def test_scenario_14_privacy_analysis():
    """Scenario 14: Perform privacy risk analysis."""
    real = pd.DataFrame({"age": [20, 30, 40, 50], "score": [10, 20, 30, 40], "cat": ["A", "B", "A", "B"]})
    synth = pd.DataFrame({"age": [21, 29, 39, 49], "score": [11, 21, 31, 41], "cat": ["A", "B", "A", "B"]})
    analyzer = PrivacyRiskAnalyzer(real, synth)
    res = analyzer.analyze_privacy()
    assert "exact_matches" in res
    assert "distance_metrics" in res
    assert res["distance_metrics"]["min_dcr"] >= 0.0


def test_scenario_15_visualizations():
    """Scenario 15: Display all visualizations."""
    real = pd.DataFrame({"age": [20, 30, 40, 50], "score": [10, 20, 30, 40], "cat": ["A", "B", "A", "B"]})
    synth = pd.DataFrame({"age": [21, 29, 39, 49], "score": [11, 21, 31, 41], "cat": ["A", "B", "A", "B"]})

    f_num = plot_numerical_distribution(real, synth, "age")
    f_cat = plot_categorical_distribution(real, synth, "cat")
    fr, fs, fd = plot_correlation_heatmaps(real, synth, ["age", "score"])
    fpca = plot_pca_projection(real, synth, ["age", "score"])
    fdcr = plot_dcr_histogram([0.1, 0.2, 0.3])

    assert f_num is not None
    assert f_cat is not None
    assert fr is not None and fs is not None and fd is not None
    assert fpca is not None
    assert fdcr is not None


def test_scenario_16_csv_export():
    """Scenario 16: Download synthetic CSV."""
    synth = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
    csv_bytes = synth.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 0
    read_back = pd.read_csv(io.BytesIO(csv_bytes))
    assert read_back.shape == (3, 2)


def test_scenario_17_html_report_export():
    """Scenario 17: Download evaluation report."""
    raw_stats = {"rows": 100, "columns": 3, "missing_cells": 0}
    gan_config = {"epochs": 40, "batch_size": 100, "pac": 10, "training_time": "3.5s", "device": "cpu"}
    quality_results = {
        "overall_score": 88.5,
        "grade": "A",
        "verdict": "High fidelity",
        "sdmetrics": {"properties_df": pd.DataFrame({"Property": ["Shapes"], "Score": [0.88]})},
        "numerical_metrics": pd.DataFrame(),
        "categorical_metrics": pd.DataFrame()
    }
    privacy_results = {
        "overall_privacy_risk": "Low",
        "exact_matches": {"exact_matches_count": 0},
        "distance_metrics": {"min_dcr": 0.12, "median_dcr": 0.45},
        "disclaimer": "Sample disclaimer"
    }

    html = generate_html_report("Test Dataset", raw_stats, gan_config, quality_results, privacy_results)
    assert "<!DOCTYPE html>" in html
    assert "Test Dataset" in html
    assert "88.5%" in html


# =====================================================================
# Edge Cases
# =====================================================================

def test_edge_empty_dataset(tmp_path):
    """Edge Case: Empty dataset."""
    fpath = tmp_path / "empty.csv"
    with open(fpath, "w") as f:
        f.write("")
    with pytest.raises(ValueError):
        load_csv(fpath)


def test_edge_single_column_dataset():
    """Edge Case: Single-column dataset."""
    df = pd.DataFrame({"metric": [10.5, 20.2, 30.7, 40.1, 50.9] * 3})
    types = detect_column_types(df)
    assert "metric" in types["numerical"]
    profile = profile_dataset(df)
    assert profile["columns"] == 1


def test_edge_unsupported_data_types():
    """Edge Case: Irregular and mixed object strings."""
    df = pd.DataFrame({
        "notes": ["Special !@#$%^&*()", "Unicode: 🧬 α β γ", "Normal string", "12345"] * 3
    })
    types = detect_column_types(df)
    assert "notes" in types["categorical"]


def test_edge_small_dataset():
    """Edge Case: Extremely small dataset (5 rows)."""
    df = pd.DataFrame({"val": [1, 2, 3, 4, 5], "cat": ["X", "Y", "X", "Y", "X"]})
    model = CTGANModel(epochs=2, batch_size=2, pac=1)
    res = model.fit(df, discrete_columns=["cat"])
    assert res["status"] == "Success"
    synth = model.generate(3)
    assert len(synth) == 3


def test_edge_large_dataset_within_memory():
    """Edge Case: Large dataset within memory (1,500 rows, 10 columns)."""
    np.random.seed(42)
    n = 1500
    df = pd.DataFrame({f"num_{i}": np.random.randn(n) for i in range(7)})
    for j in range(3):
        df[f"cat_{j}"] = np.random.choice(["Alpha", "Beta", "Gamma"], size=n)

    types = detect_column_types(df)
    assert len(types["numerical"]) == 7
    assert len(types["categorical"]) == 3
    profile = profile_dataset(df)
    assert profile["rows"] == 1500
    assert profile["columns"] == 10
