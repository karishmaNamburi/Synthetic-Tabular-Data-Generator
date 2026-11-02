"""
Unit tests for empirical privacy and disclosure risk analysis module.
"""
import pytest
import pandas as pd
from src.privacy_analyzer import PrivacyRiskAnalyzer


def test_privacy_analyzer():
    df_real = pd.DataFrame({
        "Age": [25, 30, 45, 50],
        "Salary": [50000, 60000, 90000, 110000],
        "Dept": ["IT", "HR", "Sales", "IT"]
    })
    df_synth = pd.DataFrame({
        "Age": [27, 33, 44, 48],
        "Salary": [52000, 62000, 88000, 105000],
        "Dept": ["IT", "HR", "Sales", "IT"]
    })

    analyzer = PrivacyRiskAnalyzer(df_real, df_synth)
    priv_res = analyzer.analyze_privacy()

    assert "exact_matches" in priv_res
    assert "distance_metrics" in priv_res
    assert "disclaimer" in priv_res
    assert priv_res["distance_metrics"]["min_dcr"] >= 0.0
