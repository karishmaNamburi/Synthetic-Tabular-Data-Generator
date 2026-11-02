"""
Data Quality and Statistical Fidelity Evaluation Engine.
Integrates SDMetrics Single Table Quality Reports, Kolmogorov-Smirnov tests,
Total Variation Distance (TVD), and Correlation Matrix divergence metrics.
"""
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sdmetrics.reports.single_table import QualityReport


class DataQualityEvaluator:
    """
    Evaluates synthetic tabular data quality against the real dataset.
    """

    def __init__(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, discrete_columns: List[str]):
        self.real_df = real_df.copy()
        self.synthetic_df = synthetic_df.copy()
        self.discrete_columns = [c for c in discrete_columns if c in real_df.columns and c in synthetic_df.columns]
        self.numerical_columns = [
            c for c in real_df.columns
            if c in synthetic_df.columns and c not in self.discrete_columns and pd.api.types.is_numeric_dtype(real_df[c])
        ]
        self.common_columns = [c for c in real_df.columns if c in synthetic_df.columns]

    def run_sdmetrics_report(self) -> Dict[str, Any]:
        """
        Runs the SDMetrics QualityReport on the real and synthetic data.
        """
        sdmetrics_cols = {}
        for col in self.common_columns:
            if col in self.discrete_columns or not pd.api.types.is_numeric_dtype(self.real_df[col]):
                sdmetrics_cols[col] = {"sdtype": "categorical"}
            else:
                sdmetrics_cols[col] = {"sdtype": "numerical"}

        metadata = {"columns": sdmetrics_cols}

        report = QualityReport()
        report.generate(
            self.real_df[self.common_columns],
            self.synthetic_df[self.common_columns],
            metadata,
            verbose=False
        )

        overall_score = float(report.get_score())
        properties_df = report.get_properties()

        # Extract details if available
        try:
            details_shape = report.get_details(property_name="Column Shapes")
        except Exception:
            details_shape = pd.DataFrame()

        try:
            details_pairs = report.get_details(property_name="Column Pair Trends")
        except Exception:
            details_pairs = pd.DataFrame()

        return {
            "overall_score": round(overall_score * 100, 2),
            "raw_score": overall_score,
            "properties_df": properties_df,
            "column_shapes_df": details_shape,
            "column_pairs_df": details_pairs
        }

    def evaluate_numerical_distributions(self) -> pd.DataFrame:
        """
        Calculates KS-test, Mean, Std, and relative divergence for numerical features.
        """
        results = []
        for col in self.numerical_columns:
            r_vals = self.real_df[col].dropna()
            s_vals = self.synthetic_df[col].dropna()

            if len(r_vals) == 0 or len(s_vals) == 0:
                continue

            ks_stat, ks_pval = stats.ks_2samp(r_vals, s_vals)

            r_mean, s_mean = float(r_vals.mean()), float(s_vals.mean())
            r_std, s_std = float(r_vals.std()), float(s_vals.std())
            mean_diff = abs(r_mean - s_mean)
            mean_pct_diff = round((mean_diff / (abs(r_mean) + 1e-6)) * 100, 2)

            # Similarity score: 1 - KS statistic (1.0 = identical distributions)
            sim_score = max(0.0, 1.0 - ks_stat)

            results.append({
                "Feature": col,
                "Real Mean": round(r_mean, 2),
                "Synth Mean": round(s_mean, 2),
                "Mean Diff (%)": mean_pct_diff,
                "Real Std": round(r_std, 2),
                "Synth Std": round(s_std, 2),
                "KS Statistic": round(ks_stat, 4),
                "KS p-value": round(ks_pval, 4),
                "Similarity Score": round(sim_score * 100, 2)
            })

        return pd.DataFrame(results)

    def evaluate_categorical_distributions(self) -> pd.DataFrame:
        """
        Computes Total Variation Distance (TVD) and category coverage for discrete features.
        TVD = 0.5 * sum(|P(x) - Q(x)|). 0 = identical, 1 = completely disjoint.
        Similarity Score = (1 - TVD) * 100%.
        """
        results = []
        for col in self.discrete_columns:
            r_counts = self.real_df[col].value_counts(normalize=True)
            s_counts = self.synthetic_df[col].value_counts(normalize=True)

            all_cats = list(set(r_counts.index).union(set(s_counts.index)))
            r_probs = [r_counts.get(cat, 0.0) for cat in all_cats]
            s_probs = [s_counts.get(cat, 0.0) for cat in all_cats]

            tvd = 0.5 * sum(abs(r - s) for r, s in zip(r_probs, s_probs))
            similarity_pct = round((1.0 - tvd) * 100, 2)

            # Check coverage
            real_cats = set(r_counts.index)
            synth_cats = set(s_counts.index)
            coverage_pct = round(len(real_cats.intersection(synth_cats)) / max(len(real_cats), 1) * 100, 1)

            results.append({
                "Feature": col,
                "Categories": len(all_cats),
                "Real Categories": len(real_cats),
                "Coverage (%)": coverage_pct,
                "TVD Distance": round(tvd, 4),
                "Fidelity Score (%)": similarity_pct
            })

        return pd.DataFrame(results)

    def evaluate_correlations(self) -> Dict[str, Any]:
        """
        Computes correlation matrices and their Frobenius difference / Mean Absolute Error.
        """
        if len(self.numerical_columns) < 2:
            return {"available": False, "mae": 0.0, "real_corr": pd.DataFrame(), "synth_corr": pd.DataFrame()}

        real_corr = self.real_df[self.numerical_columns].corr()
        synth_corr = self.synthetic_df[self.numerical_columns].corr()

        diff_corr = (real_corr - synth_corr).abs()
        mae_corr = float(diff_corr.values[np.triu_indices_from(diff_corr.values, k=1)].mean())

        # Correlation preservation score
        corr_score = max(0.0, 1.0 - mae_corr) * 100

        return {
            "available": True,
            "mae": round(mae_corr, 4),
            "preservation_score": round(corr_score, 2),
            "real_corr": real_corr,
            "synth_corr": synth_corr,
            "diff_corr": diff_corr
        }

    def generate_full_evaluation(self) -> Dict[str, Any]:
        """
        Executes all evaluation tests and bundles them with academic letter grading.
        """
        sdm_res = self.run_sdmetrics_report()
        num_res = self.evaluate_numerical_distributions()
        cat_res = self.evaluate_categorical_distributions()
        corr_res = self.evaluate_correlations()

        overall = sdm_res["overall_score"]
        if overall >= 85:
            grade = "A (High Fidelity)"
            verdict = "The synthetic dataset preserves statistical distributions and pairwise interactions with high accuracy."
        elif overall >= 75:
            grade = "B (Good Fidelity)"
            verdict = "The synthetic data reliably reflects the majority of patterns, with minor variance in complex interactions."
        elif overall >= 60:
            grade = "C (Moderate Fidelity)"
            verdict = "General marginal distributions are captured, but higher-order relationships show noticeable divergence."
        else:
            grade = "D (Needs Optimization)"
            verdict = "The generative model requires higher training epochs or batch size tuning to learn underlying distributions."

        return {
            "overall_score": overall,
            "grade": grade,
            "verdict": verdict,
            "sdmetrics": sdm_res,
            "numerical_metrics": num_res,
            "categorical_metrics": cat_res,
            "correlation_metrics": corr_res
        }
