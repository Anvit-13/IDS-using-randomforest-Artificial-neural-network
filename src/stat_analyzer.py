"""
StatAnalyzer Module for Network Traffic.

Computes comprehensive statistical analyses including protocol distributions, top talkers,
Shannon entropy of IP/Protocol fields, higher statistical moments (mean, median, variance,
skewness, kurtosis, 95% CI), and feature correlation matrices.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.logger import setup_logger

logger = setup_logger("StatAnalyzer")


class StatAnalyzer:
    """Class responsible for statistical analysis of packet and flow datasets."""

    def __init__(self, packet_df: pd.DataFrame, flow_df: pd.DataFrame) -> None:
        """Initializes StatAnalyzer.

        Args:
            packet_df: Cleaned packet DataFrame.
            flow_df: Extracted flow DataFrame.
        """
        self.packet_df = packet_df
        self.flow_df = flow_df
        self.results: Dict[str, Any] = {}

    def run_full_analysis(self) -> Dict[str, Any]:
        """Runs the complete statistical analysis suite.

        Returns:
            Dict[str, Any]: Dictionary containing all statistical metrics.
        """
        logger.info("Executing Statistical Analysis Suite...")

        self.results["dataset_summary"] = self._compute_summary()
        self.results["protocol_distribution"] = self._compute_protocol_distribution()
        self.results["top_source_ips"] = self._compute_top_talkers("Source", top_n=10)
        self.results["top_dest_ips"] = self._compute_top_talkers("Destination", top_n=10)
        self.results["top_source_ports"] = self._compute_top_ports("Source_Port", top_n=10)
        self.results["top_dest_ports"] = self._compute_top_ports("Destination_Port", top_n=10)
        self.results["entropy"] = self._compute_shannon_entropy()
        self.results["numerical_moments"] = self._compute_higher_moments()
        self.results["correlation_matrix"] = self._compute_correlation_matrix()

        logger.info("Statistical Analysis Completed successfully.")
        return self.results

    def _compute_summary(self) -> Dict[str, Any]:
        """Computes general summary counts and statistics."""
        packet_count = len(self.packet_df)
        flow_count = len(self.flow_df)
        total_bytes = int(self.packet_df["Length"].sum()) if "Length" in self.packet_df else 0
        duration = float(self.packet_df["Time"].max() - self.packet_df["Time"].min()) if "Time" in self.packet_df else 0.0

        return {
            "total_packets": packet_count,
            "total_flows": flow_count,
            "total_bytes": total_bytes,
            "capture_duration_seconds": round(duration, 2),
            "avg_packet_rate": round(packet_count / max(0.001, duration), 2),
            "avg_byte_rate": round(total_bytes / max(0.001, duration), 2),
        }

    def _compute_protocol_distribution(self) -> Dict[str, int]:
        """Computes distribution count per protocol."""
        if "Protocol" in self.packet_df.columns:
            return self.packet_df["Protocol"].value_counts().to_dict()
        return {}

    def _compute_top_talkers(self, col: str, top_n: int = 10) -> Dict[str, int]:
        """Computes top IP address frequencies."""
        if col in self.packet_df.columns:
            return self.packet_df[col].value_counts().head(top_n).to_dict()
        return {}

    def _compute_top_ports(self, col: str, top_n: int = 10) -> Dict[int, int]:
        """Computes top Port number frequencies."""
        if col in self.packet_df.columns:
            valid_ports = self.packet_df[self.packet_df[col] > 0][col]
            return valid_ports.value_counts().head(top_n).to_dict()
        return {}

    def _compute_shannon_entropy(self) -> Dict[str, float]:
        """Calculates Shannon Entropy (H = -sum(p * log2(p))) for Source IP, Destination IP, Protocol."""
        entropies = {}
        for col in ["Source", "Destination", "Protocol"]:
            if col in self.packet_df.columns:
                series = self.packet_df[col].astype(str)
                probs = series.value_counts(normalize=True).values
                entropy_val = -np.sum(probs * np.log2(probs + 1e-12))
                entropies[f"entropy_{col.lower()}"] = round(float(entropy_val), 4)
        return entropies

    def _compute_higher_moments(self) -> pd.DataFrame:
        """Computes Mean, Median, Variance, Skewness, Kurtosis, 95% Confidence Interval for numeric flow features."""
        numeric_cols = self.flow_df.select_dtypes(include=[np.number]).columns
        # Exclude port numbers from higher moment distribution analysis
        exclude_cols = {"Source_Port", "Destination_Port"}
        target_cols = [col for col in numeric_cols if col not in exclude_cols]

        records = []
        for col in target_cols:
            data = self.flow_df[col].dropna().values
            n = len(data)
            if n == 0:
                continue

            mean_val = float(np.mean(data))
            median_val = float(np.median(data))
            var_val = float(np.var(data, ddof=1)) if n > 1 else 0.0
            std_val = float(np.std(data, ddof=1)) if n > 1 else 0.0
            skew_val = float(stats.skew(data)) if n > 2 else 0.0
            kurt_val = float(stats.kurtosis(data)) if n > 3 else 0.0

            # 95% Confidence Interval: Mean +/- 1.96 * (std / sqrt(n))
            margin_err = 1.96 * (std_val / np.sqrt(n)) if n > 0 else 0.0
            ci_lower = mean_val - margin_err
            ci_upper = mean_val + margin_err

            records.append({
                "Feature": col,
                "Mean": round(mean_val, 4),
                "Median": round(median_val, 4),
                "Variance": round(var_val, 4),
                "Std_Dev": round(std_val, 4),
                "Skewness": round(skew_val, 4),
                "Kurtosis": round(kurt_val, 4),
                "CI_95_Lower": round(ci_lower, 4),
                "CI_95_Upper": round(ci_upper, 4),
            })

        return pd.DataFrame(records)

    def _compute_correlation_matrix(self) -> pd.DataFrame:
        """Computes Pearson correlation matrix for numeric flow features."""
        numeric_df = self.flow_df.select_dtypes(include=[np.number])
        # Drop constant columns or single-value columns
        valid_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() > 1]
        return numeric_df[valid_cols].corr().round(4)
