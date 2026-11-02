"""
Unit tests for data preprocessor and metadata generation.
"""
import pytest
import pandas as pd
from src.preprocessor import Preprocessor, build_sdv_metadata


def test_preprocessor_imputation_and_deduplication():
    df = pd.DataFrame({
        "User_ID": ["U1", "U2", "U3", "U3"],
        "Age": [25, None, 40, 40],
        "Category": ["A", "B", None, None]
    })
    detected = {"id_like": ["User_ID"], "categorical": ["Category"], "numerical": ["Age"]}

    prep = Preprocessor(impute_strategy="impute_median_mode", drop_duplicates=True, drop_id_columns=True)
    clean_df, audit = prep.fit_transform(df, detected)

    assert "User_ID" not in clean_df.columns
    assert clean_df.isna().sum().sum() == 0
    assert audit["initial_duplicates"] == 1
    assert audit["final_missing"] == 0


def test_build_sdv_metadata():
    df = pd.DataFrame({
        "Age": [20, 30, 40],
        "Department": ["IT", "HR", "Sales"]
    })
    sdv_meta, sdmetrics_meta = build_sdv_metadata(df, discrete_columns=["Department"])
    assert "columns" in sdmetrics_meta
    assert sdmetrics_meta["columns"]["Department"]["sdtype"] == "categorical"
    assert sdmetrics_meta["columns"]["Age"]["sdtype"] == "numerical"
