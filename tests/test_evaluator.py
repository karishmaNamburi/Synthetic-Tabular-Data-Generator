"""
Unit tests for data quality evaluation module.
"""
import pytest
import pandas as pd
from src.evaluator import DataQualityEvaluator


def test_data_quality_evaluator():
    df_real = pd.DataFrame({
        "Age": [25, 30, 45, 50, 35, 40],
        "Salary": [50000, 60000, 90000, 110000, 75000, 85000],
        "Dept": ["IT", "HR", "IT", "Sales", "HR", "Sales"]
    })
    df_synth = pd.DataFrame({
        "Age": [26, 31, 44, 49, 36, 41],
        "Salary": [51000, 59000, 91000, 108000, 74000, 86000],
        "Dept": ["IT", "HR", "IT", "Sales", "HR", "Sales"]
    })

    evaluator = DataQualityEvaluator(df_real, df_synth, discrete_columns=["Dept"])
    full_eval = evaluator.generate_full_evaluation()

    assert "overall_score" in full_eval
    assert full_eval["overall_score"] > 60.0
    assert not full_eval["numerical_metrics"].empty
    assert not full_eval["categorical_metrics"].empty
    assert full_eval["correlation_metrics"]["available"]
