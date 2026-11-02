"""
Data Preprocessing and SDV/SDMetrics Metadata extraction module.
"""
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sdv.metadata import Metadata


class Preprocessor:
    """
    Cleans tabular data, manages missing values and duplicates,
    and produces metadata compliant with SDV and SDMetrics.
    """

    def __init__(
        self,
        impute_strategy: str = "impute_median_mode",  # "impute_median_mode", "impute_mean_mode", "drop_rows", "none"
        drop_duplicates: bool = True,
        drop_id_columns: bool = True
    ):
        self.impute_strategy = impute_strategy
        self.drop_duplicates_flag = drop_duplicates
        self.drop_id_columns_flag = drop_id_columns
        self.imputation_values_: Dict[str, Any] = {}
        self.dropped_cols_: List[str] = []
        self.audit_log_: Dict[str, Any] = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        column_types: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fits preprocessing parameters and transforms the dataframe.
        Returns cleaned DataFrame and an audit summary.
        """
        df_clean = df.copy()
        initial_rows, initial_cols = df_clean.shape
        initial_missing = int(df_clean.isna().sum().sum())
        initial_duplicates = int(df_clean.duplicated().sum())

        # 1. Handle ID columns
        if column_types and self.drop_id_columns_flag:
            id_cols = [c for c in column_types.get("id_like", []) if c in df_clean.columns]
            if id_cols:
                df_clean = df_clean.drop(columns=id_cols)
                self.dropped_cols_ = id_cols

        # 2. Handle Duplicates
        if self.drop_duplicates_flag and df_clean.duplicated().any():
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)

        # 3. Handle Missing Values
        if self.impute_strategy == "drop_rows":
            df_clean = df_clean.dropna().reset_index(drop=True)
        elif self.impute_strategy in ["impute_median_mode", "impute_mean_mode"]:
            for col in df_clean.columns:
                if df_clean[col].isna().any():
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        if self.impute_strategy == "impute_median_mode":
                            val = df_clean[col].median()
                        else:
                            val = df_clean[col].mean()
                        if pd.isna(val):
                            val = 0
                        self.imputation_values_[col] = val
                        df_clean[col] = df_clean[col].fillna(val)
                    else:
                        # Categorical: use mode
                        mode_series = df_clean[col].mode(dropna=True)
                        val = mode_series.iloc[0] if not mode_series.empty else "Missing"
                        self.imputation_values_[col] = val
                        df_clean[col] = df_clean[col].fillna(val)

        final_rows, final_cols = df_clean.shape
        final_missing = int(df_clean.isna().sum().sum())
        final_duplicates = int(df_clean.duplicated().sum())

        audit = {
            "initial_shape": (initial_rows, initial_cols),
            "final_shape": (final_rows, final_cols),
            "initial_missing": initial_missing,
            "final_missing": final_missing,
            "initial_duplicates": initial_duplicates,
            "final_duplicates": final_duplicates,
            "dropped_columns": self.dropped_cols_,
            "imputed_columns": list(self.imputation_values_.keys()),
            "imputation_strategy": self.impute_strategy
        }
        self.audit_log_ = audit
        return df_clean, audit


def build_sdv_metadata(df: pd.DataFrame, discrete_columns: List[str]) -> Tuple[Metadata, Dict[str, Any]]:
    """
    Constructs both an SDV Metadata object and a pure SDMetrics metadata dict.
    Ensures strict type compliance for evaluation.
    """
    sdmetrics_columns = {}
    for col in df.columns:
        if col in discrete_columns or not pd.api.types.is_numeric_dtype(df[col]):
            sdmetrics_columns[col] = {"sdtype": "categorical"}
        else:
            sdmetrics_columns[col] = {"sdtype": "numerical"}

    sdmetrics_meta = {"columns": sdmetrics_columns}

    # SDV 2.x Metadata object
    sdv_meta = Metadata()
    sdv_meta.detect_from_dataframes(data={"main_table": df})

    return sdv_meta, sdmetrics_meta
