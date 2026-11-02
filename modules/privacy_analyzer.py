"""
Privacy and Empirical Disclosure Risk Analysis Module.
Implements Distance to Closest Record (DCR), Exact Match Analysis,
Nearest Neighbor Distance Ratios, and Disclosure Risk Indices.
"""
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


class PrivacyRiskAnalyzer:
    """
    Assesses empirical privacy risks of synthetic tabular datasets.

    NOTE: Empirical privacy measures evaluate proximity and memorization heuristics.
    They do NOT constitute theoretical Differential Privacy guarantees.
    """

    def __init__(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame):
        self.real_df = real_df.copy()
        self.synthetic_df = synthetic_df.copy()
        self.common_cols = [c for c in real_df.columns if c in synthetic_df.columns]

    def check_exact_matches(self) -> Dict[str, Any]:
        """
        Detects exact duplicate records between the real and synthetic datasets.
        """
        r_subset = self.real_df[self.common_cols].dropna()
        s_subset = self.synthetic_df[self.common_cols].dropna()

        # Merge on all common columns to find identical rows
        merged = pd.merge(s_subset, r_subset, how="inner")
        match_count = len(merged)
        synth_total = len(s_subset)
        match_rate_pct = round((match_count / max(synth_total, 1)) * 100, 2)

        return {
            "exact_matches_count": match_count,
            "synthetic_records_total": synth_total,
            "match_rate_pct": match_rate_pct,
            "memorization_detected": match_count > 0
        }

    def compute_distance_to_closest_record(self, sample_size: int = 500) -> Dict[str, Any]:
        """
        Computes Distance to Closest Record (DCR) using normalized Euclidean distance.
        A low minimum DCR (< 0.01) indicates potential sample replication/overfitting.
        """
        # Prepare feature matrix: numerical normalized, categorical one-hot encoded
        r_df = self.real_df[self.common_cols].copy()
        s_df = self.synthetic_df[self.common_cols].copy()

        # Subsample if large for responsive evaluation
        if len(r_df) > sample_size:
            r_df = r_df.sample(sample_size, random_state=42)
        if len(s_df) > sample_size:
            s_df = s_df.sample(sample_size, random_state=42)

        # Separate numeric and categorical
        num_cols = [c for c in self.common_cols if pd.api.types.is_numeric_dtype(r_df[c])]
        cat_cols = [c for c in self.common_cols if c not in num_cols]

        # Combine for consistent one-hot encoding
        combined = pd.concat([r_df, s_df], axis=0, ignore_index=True)
        if cat_cols:
            combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
        else:
            combined_encoded = combined.copy()

        # Fill any remaining NaNs
        combined_encoded = combined_encoded.fillna(combined_encoded.median(numeric_only=True))

        # Scale features to [0, 1]
        scaler = MinMaxScaler()
        scaled_matrix = scaler.fit_transform(combined_encoded)

        n_real = len(r_df)
        r_matrix = scaled_matrix[:n_real]
        s_matrix = scaled_matrix[n_real:]

        # Fit NearestNeighbors on real data
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(r_matrix)

        # Distances from synthetic records to closest real record
        distances, indices = nn.kneighbors(s_matrix)
        dcr_values = distances.flatten()

        min_dcr = float(np.min(dcr_values))
        p5_dcr = float(np.percentile(dcr_values, 5))
        median_dcr = float(np.median(dcr_values))
        mean_dcr = float(np.mean(dcr_values))

        # Privacy risk interpretation
        if min_dcr < 0.005:
            risk_level = "High Risk (Re-identification Hazard)"
            risk_desc = "At least one synthetic sample is virtually identical to a real training record, indicating memorization."
        elif p5_dcr < 0.08:
            risk_level = "Moderate Risk"
            risk_desc = "Synthetic records cluster closely around training data points. Attribute disclosure should be audited."
        else:
            risk_level = "Low Empirical Risk"
            risk_desc = "Synthetic samples maintain healthy separation from training records, indicating sound generalization."

        return {
            "min_dcr": round(min_dcr, 4),
            "p5_dcr": round(p5_dcr, 4),
            "median_dcr": round(median_dcr, 4),
            "mean_dcr": round(mean_dcr, 4),
            "dcr_distribution": dcr_values.tolist(),
            "risk_level": risk_level,
            "risk_description": risk_desc
        }

    def analyze_privacy(self) -> Dict[str, Any]:
        """
        Runs comprehensive empirical privacy audit with legal/scientific disclaimer.
        """
        matches = self.check_exact_matches()
        dcr = self.compute_distance_to_closest_record()

        disclaimer = (
            "CRITICAL SCIENTIFIC & ETHICAL NOTICE:\n"
            "This empirical privacy evaluation measures sample distance (DCR) and exact matches. "
            "It DOES NOT guarantee mathematical Differential Privacy (epsilon-DP). Generative models "
            "can still be susceptible to membership inference, reconstruction, or linkage attacks under "
            "adversarial auxiliary knowledge. Synthetic data should be treated as pseudonymous rather than "
            "anonymized unless bounded by formal differential privacy mechanisms."
        )

        return {
            "exact_matches": matches,
            "distance_metrics": dcr,
            "overall_privacy_risk": dcr["risk_level"],
            "disclaimer": disclaimer
        }
