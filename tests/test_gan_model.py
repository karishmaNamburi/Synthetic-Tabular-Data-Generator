"""
Unit and integration tests for CTGAN model training and generation.
"""
import pytest
import pandas as pd
from src.gan_model import CTGANModel


def test_ctgan_fit_and_generate():
    df = pd.DataFrame({
        "Age": [22, 28, 35, 42, 50, 58, 26, 31, 45, 52],
        "Salary": [30000, 45000, 60000, 80000, 95000, 110000, 38000, 52000, 85000, 98000],
        "Department": ["IT", "HR", "IT", "Sales", "HR", "Sales", "IT", "Sales", "HR", "IT"]
    })

    model = CTGANModel(epochs=2, batch_size=10, pac=2, verbose=False)
    fit_res = model.fit(df, discrete_columns=["Department"])

    assert model.is_trained
    assert fit_res["status"] == "Success"

    synthetic_df = model.generate(num_rows=6, preserve_types=True)
    assert len(synthetic_df) == 6
    assert list(synthetic_df.columns) == list(df.columns)
    assert (synthetic_df["Age"] >= 0).all()
