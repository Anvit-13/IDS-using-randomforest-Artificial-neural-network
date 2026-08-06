"""
ReportGenerator Module for Network Traffic Analysis.

Generates and exports structured reports (CSV, TXT, JSON, HTML, Markdown) summarizing packet,
flow, protocol, statistical, and machine learning anomaly metrics.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import pandas as pd

from src.logger import setup_logger

logger = setup_logger("ReportGenerator")


class ReportGenerator:
    """Class responsible for formatting and exporting pipeline results into multiple formats."""

    def __init__(
        self,
        reports_dir: Path = Path("data/reports"),
        processed_dir: Path = Path("data/processed"),
    ) -> None:
        """Initializes ReportGenerator.

        Args:
            reports_dir: Directory for storing output reports.
            processed_dir: Directory for storing clean processed datasets.
        """
        self.reports_dir = Path(reports_dir)
        self.processed_dir = Path(processed_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def export_processed_datasets(
        self,
        clean_packets_df: pd.DataFrame,
        flow_features_df: pd.DataFrame,
    ) -> Tuple[Path, Path]:
        """Saves clean_packets.csv and flow_features.csv to processed directory.

        Args:
            clean_packets_df: Cleaned packet DataFrame.
            flow_features_df: GAN-ready flow features DataFrame.

        Returns:
            Tuple[Path, Path]: Paths to clean_packets.csv and flow_features.csv.
        """
        packets_path = self.processed_dir / "clean_packets.csv"
        flows_path = self.processed_dir / "flow_features.csv"

        clean_packets_df.to_csv(packets_path, index=False)
        flow_features_df.to_csv(flows_path, index=False)

        logger.info(f"Saved cleaned packet capture to '{packets_path}'.")
        logger.info(f"Saved GAN-ready flow features dataset to '{flows_path}'.")
        return packets_path, flows_path

    def generate_all_reports(
        self,
        stat_results: Dict[str, Any],
        cleaning_report: Dict[str, Any],
        anomaly_df: pd.DataFrame,
    ) -> None:
        """Generates all CSV, TXT, JSON, HTML, and Markdown summary reports.

        Args:
            stat_results: Output dictionary from StatAnalyzer.
            cleaning_report: Output dictionary from DataCleaner.
            anomaly_df: Anomaly predictions DataFrame from MLPipeline.
        """
        logger.info("Generating and exporting pipeline reports...")

        # 1. Packet & Protocol Statistics CSVs
        self._export_statistics_csvs(stat_results)

        # 2. Anomaly Report CSV
        anomaly_path = self.reports_dir / "anomaly_report.csv"
        anomaly_df.to_csv(anomaly_path, index=False)
        logger.info(f"Saved anomaly evaluation report to '{anomaly_path}'.")

        # 3. Summary JSON
        summary_json = self._build_summary_json(stat_results, cleaning_report, anomaly_df)
        json_path = self.reports_dir / "summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=4)
        logger.info(f"Saved summary JSON to '{json_path}'.")

        # 4. Text Summary Report
        self._export_txt_report(summary_json)

        # 5. Markdown Report (Bonus)
        self._export_markdown_report(summary_json)

        # 6. HTML Report (Bonus)
        self._export_html_report(summary_json)

        logger.info("All pipeline reports generated successfully.")

    def _export_statistics_csvs(self, stat_results: Dict[str, Any]) -> None:
        """Exports packet_statistics.csv and protocol_statistics.csv."""
        # Packet higher moments dataframe
        moments_df = stat_results.get("numerical_moments")
        if moments_df is not None and not moments_df.empty:
            moments_path = self.reports_dir / "packet_statistics.csv"
            moments_df.to_csv(moments_path, index=False)
            logger.info(f"Saved packet statistical moments to '{moments_path}'.")

        # Protocol statistics dataframe
        proto_dist = stat_results.get("protocol_distribution", {})
        if proto_dist:
            proto_df = pd.DataFrame(list(proto_dict.items()), columns=["Protocol", "Packet_Count"]) if (proto_dict := proto_dist) else pd.DataFrame()
            proto_df["Percentage"] = (proto_df["Packet_Count"] / proto_df["Packet_Count"].sum() * 100).round(2)
            proto_path = self.reports_dir / "protocol_statistics.csv"
            proto_df.to_csv(proto_path, index=False)
            logger.info(f"Saved protocol statistics to '{proto_path}'.")

    def _build_summary_json(
        self,
        stat_results: Dict[str, Any],
        cleaning_report: Dict[str, Any],
        anomaly_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Constructs comprehensive nested dictionary for summary JSON export."""
        summary = stat_results.get("dataset_summary", {}).copy()
        summary["data_cleaning"] = cleaning_report
        summary["entropy_metrics"] = stat_results.get("entropy", {})
        summary["top_protocols"] = stat_results.get("protocol_distribution", {})
        summary["top_source_ips"] = stat_results.get("top_source_ips", {})
        summary["top_destination_ips"] = stat_results.get("top_dest_ips", {})
        summary["top_source_ports"] = stat_results.get("top_source_ports", {})
        summary["top_destination_ports"] = stat_results.get("top_dest_ports", {})

        # Anomaly Summary
        total_flows = len(anomaly_df)
        iso_anomalies = int((anomaly_df.get("IsoForest_Pred", pd.Series()) == -1).sum())
        ocsvm_anomalies = int((anomaly_df.get("OneClassSVM_Pred", pd.Series()) == -1).sum())
        dbscan_anomalies = int((anomaly_df.get("DBSCAN_Anomaly", pd.Series()) == -1).sum())
        consensus_anomalies = int((anomaly_df.get("Consensus_Anomaly", pd.Series()) == -1).sum())

        summary["anomaly_detection_summary"] = {
            "total_flows_evaluated": total_flows,
            "isolation_forest_anomalies": iso_anomalies,
            "one_class_svm_anomalies": ocsvm_anomalies,
            "dbscan_outliers": dbscan_anomalies,
            "consensus_anomalies": consensus_anomalies,
            "anomaly_rate_percent": round((consensus_anomalies / max(1, total_flows)) * 100, 2),
        }

        return summary

    def _export_txt_report(self, summary: Dict[str, Any]) -> None:
        """Saves human-readable plain text summary report."""
        txt_path = self.reports_dir / "summary.txt"
        lines = [
            "================================================================================",
            "  NETWORK TRAFFIC ANALYSIS & FEATURE EXTRACTION PIPELINE - SUMMARY REPORT",
            "================================================================================",
            "",
            "1. DATASET OVERVIEW",
            "--------------------------------------------------------------------------------",
            f"  - Total Packets Processed: {summary.get('total_packets', 0):,}",
            f"  - Total Network Flows:      {summary.get('total_flows', 0):,}",
            f"  - Total Data Volume:       {summary.get('total_bytes', 0):,} Bytes",
            f"  - Capture Duration:        {summary.get('capture_duration_seconds', 0)} Seconds",
            f"  - Avg Packet Rate:         {summary.get('avg_packet_rate', 0)} Packets/sec",
            f"  - Avg Byte Rate:           {summary.get('avg_byte_rate', 0)} Bytes/sec",
            "",
            "2. DATA CLEANING AUDIT",
            "--------------------------------------------------------------------------------",
            f"  - Initial Rows:            {summary.get('data_cleaning', {}).get('initial_row_count', 0):,}",
            f"  - Duplicates Removed:      {summary.get('data_cleaning', {}).get('duplicate_rows_removed', 0):,}",
            f"  - Malformed Rows Dropped:  {summary.get('data_cleaning', {}).get('malformed_rows_removed', 0):,}",
            f"  - Missing Values Handled:  {summary.get('data_cleaning', {}).get('missing_values_handled', 0):,}",
            f"  - Final Clean Packets:     {summary.get('data_cleaning', {}).get('final_row_count', 0):,}",
            "",
            "3. SHANNON ENTROPY METRICS",
            "--------------------------------------------------------------------------------",
        ]

        for k, v in summary.get("entropy_metrics", {}).items():
            lines.append(f"  - {k.replace('_', ' ').title()}: {v}")

        lines.extend([
            "",
            "4. ANOMALY DETECTION SUMMARY",
            "--------------------------------------------------------------------------------",
            f"  - Total Flows Analyzed:    {summary.get('anomaly_detection_summary', {}).get('total_flows_evaluated', 0):,}",
            f"  - Isolation Forest Hits:   {summary.get('anomaly_detection_summary', {}).get('isolation_forest_anomalies', 0):,}",
            f"  - One-Class SVM Hits:      {summary.get('anomaly_detection_summary', {}).get('one_class_svm_anomalies', 0):,}",
            f"  - DBSCAN Outliers:         {summary.get('anomaly_detection_summary', {}).get('dbscan_outliers', 0):,}",
            f"  - Consensus Anomalies:     {summary.get('anomaly_detection_summary', {}).get('consensus_anomalies', 0):,}",
            f"  - Overall Anomaly Rate:    {summary.get('anomaly_detection_summary', {}).get('anomaly_rate_percent', 0)}%",
            "",
            "================================================================================",
        ])

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved text report to '{txt_path}'.")

    def _export_markdown_report(self, summary: Dict[str, Any]) -> None:
        """Saves rich Markdown report with summary tables."""
        md_path = self.reports_dir / "summary.md"
        md_content = f"""# Network Traffic Analysis & GAN Feature Pipeline Report

## 1. Executive Summary
- **Total Packets:** {summary.get('total_packets', 0):,}
- **Total Flows:** {summary.get('total_flows', 0):,}
- **Total Bytes:** {summary.get('total_bytes', 0):,}
- **Capture Duration:** {summary.get('capture_duration_seconds', 0)} s

## 2. Shannon Entropy Metrics
| Field | Shannon Entropy (bits) |
|---|---|
| Source IP | {summary.get('entropy_metrics', {}).get('entropy_source', 'N/A')} |
| Destination IP | {summary.get('entropy_metrics', {}).get('entropy_destination', 'N/A')} |
| Protocol | {summary.get('entropy_metrics', {}).get('entropy_protocol', 'N/A')} |

## 3. Anomaly Detection Summary
| Model | Anomalies Flagged | Percentage |
|---|---|---|
| Isolation Forest | {summary.get('anomaly_detection_summary', {}).get('isolation_forest_anomalies', 0):,} | - |
| One-Class SVM | {summary.get('anomaly_detection_summary', {}).get('one_class_svm_anomalies', 0):,} | - |
| DBSCAN Outliers | {summary.get('anomaly_detection_summary', {}).get('dbscan_outliers', 0):,} | - |
| **Consensus Anomalies** | **{summary.get('anomaly_detection_summary', {}).get('consensus_anomalies', 0):,}** | **{summary.get('anomaly_detection_summary', {}).get('anomaly_rate_percent', 0)}%** |

---
*Report generated automatically by IDS GAN Feature Extraction Pipeline.*
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Saved Markdown report to '{md_path}'.")

    def _export_html_report(self, summary: Dict[str, Any]) -> None:
        """Saves styled standalone HTML report."""
        html_path = self.reports_dir / "summary.html"
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IDS GAN Pipeline Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background-color: #f8f9fa; color: #212529; }}
        h1 {{ color: #0d6efd; border-bottom: 2px solid #0d6efd; padding-bottom: 10px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; border: 1px solid #dee2e6; text-align: left; }}
        th {{ background-color: #e9ecef; }}
    </style>
</head>
<body>
    <h1>Network Traffic Analysis & Feature Extraction Report</h1>
    
    <div class="card">
        <h2>Dataset Overview</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Packets</td><td>{summary.get('total_packets', 0):,}</td></tr>
            <tr><td>Total Flows</td><td>{summary.get('total_flows', 0):,}</td></tr>
            <tr><td>Total Bytes</td><td>{summary.get('total_bytes', 0):,}</td></tr>
            <tr><td>Capture Duration</td><td>{summary.get('capture_duration_seconds', 0)} s</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>Anomaly Detection Summary</h2>
        <table>
            <tr><th>Detector</th><th>Count</th></tr>
            <tr><td>Isolation Forest</td><td>{summary.get('anomaly_detection_summary', {}).get('isolation_forest_anomalies', 0):,}</td></tr>
            <tr><td>One-Class SVM</td><td>{summary.get('anomaly_detection_summary', {}).get('one_class_svm_anomalies', 0):,}</td></tr>
            <tr><td>DBSCAN Noise</td><td>{summary.get('anomaly_detection_summary', {}).get('dbscan_outliers', 0):,}</td></tr>
            <tr><td><strong>Majority Consensus</strong></td><td><strong>{summary.get('anomaly_detection_summary', {}).get('consensus_anomalies', 0):,} ({summary.get('anomaly_detection_summary', {}).get('anomaly_rate_percent', 0)}%)</strong></td></tr>
        </table>
    </div>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved HTML report to '{html_path}'.")
