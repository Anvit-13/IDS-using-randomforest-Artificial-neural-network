"""
Visualizer Module for Network Traffic Analysis.

Generates publication-quality high-resolution figures (300 DPI) in both PNG and SVG formats.
Includes scatter plots, histograms, boxplots, bar charts, heatmaps, pie charts, and time series line graphs.
"""

from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.logger import setup_logger

logger = setup_logger("Visualizer")


class Visualizer:
    """Class responsible for generating publication-quality visualizations."""

    def __init__(
        self,
        output_dir: Path = Path("data/plots"),
        dpi: int = 300,
        palette: str = "deep",
    ) -> None:
        """Initializes Visualizer.

        Args:
            output_dir: Directory where figures will be saved.
            dpi: Dots per inch for raster PNG saving (default 300).
            palette: Seaborn color palette name.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.palette = palette

        # Configure aesthetic theme
        sns.set_theme(style="whitegrid", palette=self.palette)
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["figure.autolayout"] = True

    def _save_fig(self, filename_base: str) -> None:
        """Saves current matplotlib figure as both 300 DPI PNG and SVG.

        Args:
            filename_base: Base filename without extension (e.g., 'packet_length_vs_time').
        """
        png_path = self.output_dir / f"{filename_base}.png"
        svg_path = self.output_dir / f"{filename_base}.svg"

        plt.savefig(png_path, dpi=self.dpi, bbox_inches="tight")
        plt.savefig(svg_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plots: '{png_path.name}' and '{svg_path.name}'")

    def generate_all_plots(
        self,
        packet_df: pd.DataFrame,
        flow_df: pd.DataFrame,
        stat_results: Dict,
    ) -> None:
        """Generates all 14 publication-quality figures required by the pipeline.

        Args:
            packet_df: Cleaned packet DataFrame.
            flow_df: Extracted flow DataFrame.
            stat_results: Statistical analysis results dictionary.
        """
        logger.info("Generating publication-quality visualization figures...")

        self.plot_scatter_packet_length_vs_time(packet_df)
        self.plot_scatter_flow_duration_vs_bytes(flow_df)
        self.plot_histogram_packet_size(packet_df)
        self.plot_histogram_flow_duration(flow_df)
        self.plot_histogram_iat(packet_df)
        self.plot_boxplot_packet_length(packet_df)
        self.plot_boxplot_flow_duration(flow_df)
        self.plot_barchart_protocol_distribution(packet_df)
        self.plot_barchart_top_source_ips(stat_results.get("top_source_ips", {}))
        self.plot_barchart_top_dest_ips(stat_results.get("top_dest_ips", {}))
        self.plot_heatmap_correlation(stat_results.get("correlation_matrix"))
        self.plot_piechart_protocol_distribution(stat_results.get("protocol_distribution", {}))
        self.plot_linegraph_packets_per_second(packet_df)
        self.plot_linegraph_bytes_per_second(packet_df)

        logger.info("All 14 visualization figures generated successfully.")

    def plot_scatter_packet_length_vs_time(self, packet_df: pd.DataFrame) -> None:
        """1. Scatter Plot: Packet Length vs Time."""
        plt.figure(figsize=(10, 6))
        # Sample for plotting speed if packet count is huge
        sample_df = packet_df.sample(min(20000, len(packet_df)), random_state=42) if len(packet_df) > 20000 else packet_df
        sns.scatterplot(data=sample_df, x="Time", y="Length", hue="Protocol", alpha=0.6, s=15)
        plt.title("Packet Length vs. Time", fontsize=14, fontweight="bold")
        plt.xlabel("Timestamp (Seconds)", fontsize=12)
        plt.ylabel("Packet Length (Bytes)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Protocol")
        self._save_fig("scatter_packet_length_vs_time")

    def plot_scatter_flow_duration_vs_bytes(self, flow_df: pd.DataFrame) -> None:
        """2. Scatter Plot: Flow Duration vs Total Bytes."""
        plt.figure(figsize=(10, 6))
        sample_df = flow_df.sample(min(10000, len(flow_df)), random_state=42) if len(flow_df) > 10000 else flow_df
        sns.scatterplot(data=sample_df, x="Flow_Duration", y="Total_Bytes", hue="Protocol", alpha=0.7, s=25)
        plt.title("Flow Duration vs. Total Bytes", fontsize=14, fontweight="bold")
        plt.xlabel("Flow Duration (Seconds)", fontsize=12)
        plt.ylabel("Total Bytes", fontsize=12)
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Protocol")
        self._save_fig("scatter_flow_duration_vs_bytes")

    def plot_histogram_packet_size(self, packet_df: pd.DataFrame) -> None:
        """3. Histogram: Packet Size Distribution."""
        plt.figure(figsize=(10, 6))
        sns.histplot(packet_df["Length"], bins=50, kde=True, color="#2b5c8f")
        plt.title("Packet Size Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Packet Length (Bytes)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        self._save_fig("histogram_packet_size")

    def plot_histogram_flow_duration(self, flow_df: pd.DataFrame) -> None:
        """4. Histogram: Flow Duration Distribution."""
        plt.figure(figsize=(10, 6))
        sns.histplot(flow_df["Flow_Duration"], bins=50, kde=True, color="#d95f02", log_scale=True)
        plt.title("Flow Duration Distribution (Log Scale)", fontsize=14, fontweight="bold")
        plt.xlabel("Flow Duration (Seconds)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        self._save_fig("histogram_flow_duration")

    def plot_histogram_iat(self, packet_df: pd.DataFrame) -> None:
        """5. Histogram: Inter-Arrival Time (IAT) Distribution."""
        plt.figure(figsize=(10, 6))
        valid_iat = packet_df[packet_df["IAT"] > 0]["IAT"]
        sns.histplot(valid_iat, bins=50, kde=True, color="#7570b3", log_scale=True)
        plt.title("Inter-Arrival Time (IAT) Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("IAT (Seconds)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        self._save_fig("histogram_inter_arrival_time")

    def plot_boxplot_packet_length(self, packet_df: pd.DataFrame) -> None:
        """6. Boxplot: Packet Length by Protocol."""
        plt.figure(figsize=(10, 6))
        top_protos = packet_df["Protocol"].value_counts().head(6).index
        filtered = packet_df[packet_df["Protocol"].isin(top_protos)]
        sns.boxplot(data=filtered, x="Protocol", y="Length", hue="Protocol", palette="Set2", legend=False)
        plt.title("Packet Length Boxplot by Top Protocols", fontsize=14, fontweight="bold")
        plt.xlabel("Protocol", fontsize=12)
        plt.ylabel("Packet Length (Bytes)", fontsize=12)
        self._save_fig("boxplot_packet_length")

    def plot_boxplot_flow_duration(self, flow_df: pd.DataFrame) -> None:
        """7. Boxplot: Flow Duration by Protocol."""
        plt.figure(figsize=(10, 6))
        top_protos = flow_df["Protocol"].value_counts().head(6).index
        filtered = flow_df[flow_df["Protocol"].isin(top_protos)]
        sns.boxplot(data=filtered, x="Protocol", y="Flow_Duration", hue="Protocol", palette="Set3", legend=False)
        plt.yscale("log")
        plt.title("Flow Duration Boxplot by Top Protocols (Log Scale)", fontsize=14, fontweight="bold")
        plt.xlabel("Protocol", fontsize=12)
        plt.ylabel("Flow Duration (Seconds)", fontsize=12)
        self._save_fig("boxplot_flow_duration")

    def plot_barchart_protocol_distribution(self, packet_df: pd.DataFrame) -> None:
        """8. Bar Chart: Protocol Distribution."""
        plt.figure(figsize=(10, 6))
        counts = packet_df["Protocol"].value_counts().head(10)
        ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="crest", legend=False)
        plt.title("Top Protocol Counts", fontsize=14, fontweight="bold")
        plt.xlabel("Protocol", fontsize=12)
        plt.ylabel("Packet Count", fontsize=12)
        plt.xticks(rotation=45)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
        self._save_fig("barchart_protocol_distribution")

    def plot_barchart_top_source_ips(self, top_src_ips: Dict[str, int]) -> None:
        """9. Bar Chart: Top Source IPs."""
        plt.figure(figsize=(10, 6))
        if not top_src_ips:
            return
        ips = list(top_src_ips.keys())
        counts = list(top_src_ips.values())
        ax = sns.barplot(x=counts, y=ips, hue=ips, palette="Blues_r", legend=False)
        plt.title("Top 10 Source IP Addresses", fontsize=14, fontweight="bold")
        plt.xlabel("Packet Count", fontsize=12)
        plt.ylabel("Source IP", fontsize=12)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{int(width):,}", (width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=10, xytext=(5, 0), textcoords='offset points')
        self._save_fig("barchart_top_source_ips")

    def plot_barchart_top_dest_ips(self, top_dst_ips: Dict[str, int]) -> None:
        """10. Bar Chart: Top Destination IPs."""
        plt.figure(figsize=(10, 6))
        if not top_dst_ips:
            return
        ips = list(top_dst_ips.keys())
        counts = list(top_dst_ips.values())
        ax = sns.barplot(x=counts, y=ips, hue=ips, palette="Purples_r", legend=False)
        plt.title("Top 10 Destination IP Addresses", fontsize=14, fontweight="bold")
        plt.xlabel("Packet Count", fontsize=12)
        plt.ylabel("Destination IP", fontsize=12)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{int(width):,}", (width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=10, xytext=(5, 0), textcoords='offset points')
        self._save_fig("barchart_top_destination_ips")

    def plot_heatmap_correlation(self, corr_df: Optional[pd.DataFrame]) -> None:
        """11. Heatmap: Correlation Matrix."""
        if corr_df is None or corr_df.empty:
            return
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_df, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
        plt.title("Flow Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        self._save_fig("heatmap_correlation_matrix")

    def plot_piechart_protocol_distribution(self, proto_dict: Dict[str, int]) -> None:
        """12. Pie Chart: Protocol Distribution."""
        if not proto_dict:
            return
        plt.figure(figsize=(8, 8))
        series = pd.Series(proto_dict)
        top = series.head(5)
        other_sum = series.iloc[5:].sum()
        if other_sum > 0:
            top["OTHER"] = other_sum

        plt.pie(top.values, labels=top.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
        plt.title("Protocol Share Breakdown", fontsize=14, fontweight="bold")
        self._save_fig("piechart_protocol_distribution")

    def plot_linegraph_packets_per_second(self, packet_df: pd.DataFrame) -> None:
        """13. Line Graph: Packets Per Second Over Time."""
        plt.figure(figsize=(12, 6))
        # Resample packets per 1-second bins
        time_series = packet_df.copy()
        time_series["Time_Int"] = time_series["Time"].astype(int)
        pps = time_series.groupby("Time_Int").size()

        plt.plot(pps.index, pps.values, color="#1f77b4", linewidth=1.5)
        plt.title("Network Traffic Intensity: Packets Per Second (PPS)", fontsize=14, fontweight="bold")
        plt.xlabel("Timeline (Seconds)", fontsize=12)
        plt.ylabel("Packets Per Second", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        self._save_fig("linegraph_packets_per_second")

    def plot_linegraph_bytes_per_second(self, packet_df: pd.DataFrame) -> None:
        """14. Line Graph: Bytes Per Second Over Time."""
        plt.figure(figsize=(12, 6))
        time_series = packet_df.copy()
        time_series["Time_Int"] = time_series["Time"].astype(int)
        bps = time_series.groupby("Time_Int")["Length"].sum()

        plt.plot(bps.index, bps.values / 1024.0, color="#2ca02c", linewidth=1.5)
        plt.title("Network Throughput: Kilobytes Per Second (KBps)", fontsize=14, fontweight="bold")
        plt.xlabel("Timeline (Seconds)", fontsize=12)
        plt.ylabel("Data Rate (KB/s)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        self._save_fig("linegraph_bytes_per_second")

    def generate_investigation_plots(
        self,
        investigation_df: pd.DataFrame,
        anomaly_df: pd.DataFrame,
        inv_summary: Dict,
    ) -> None:
        """Generates all Flow Investigation visualization figures.

        Args:
            investigation_df: DataFrame containing consensus anomalous flows.
            anomaly_df: Full ML prediction DataFrame.
            inv_summary: Investigation summary dictionary.
        """
        logger.info("Generating Flow Investigation visualization figures...")
        if investigation_df.empty:
            logger.warning("Investigation DataFrame is empty. Skipping investigation plots.")
            return

        self.plot_investigation_anomalous_flows_by_protocol(inv_summary.get("anomalous_flows_by_protocol", {}))
        self.plot_investigation_anomalous_packets_by_protocol(inv_summary.get("anomalous_packets_by_protocol", {}))
        self.plot_investigation_top_source_ips(inv_summary.get("top_anomalous_source_ips", {}))
        self.plot_investigation_top_dest_ips(inv_summary.get("top_anomalous_destination_ips", {}))
        self.plot_investigation_category_distribution(inv_summary.get("most_common_possible_categories", {}))
        self.plot_investigation_behaviour_distribution(inv_summary.get("most_common_behaviours", {}))
        self.plot_investigation_model_agreement(anomaly_df)

        logger.info("All Flow Investigation plots generated successfully.")

    def plot_investigation_anomalous_flows_by_protocol(self, proto_dict: Dict[str, int]) -> None:
        """Investigation Plot 1: Anomalous Flows by Protocol."""
        if not proto_dict:
            return
        plt.figure(figsize=(10, 6))
        series = pd.Series(proto_dict).head(10)
        ax = sns.barplot(x=series.values, y=series.index, hue=series.index, palette="Reds_r", legend=False)
        plt.title("Consensus Anomalous Flows by Protocol", fontsize=14, fontweight="bold")
        plt.xlabel("Anomalous Flow Count", fontsize=12)
        plt.ylabel("Protocol", fontsize=12)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{int(width):,}", (width, p.get_y() + p.get_height() / 2.),
                        ha="left", va="center", fontsize=10, xytext=(5, 0), textcoords="offset points")
        self._save_fig("investigation_anomalous_flows_by_protocol")

    def plot_investigation_anomalous_packets_by_protocol(self, proto_pkts_dict: Dict[str, int]) -> None:
        """Investigation Plot 2: Packets in Anomalous Flows by Protocol."""
        if not proto_pkts_dict:
            return
        plt.figure(figsize=(10, 6))
        series = pd.Series(proto_pkts_dict).head(10)
        ax = sns.barplot(x=series.values, y=series.index, hue=series.index, palette="Oranges_r", legend=False)
        plt.title("Anomalous Flow Packets by Protocol", fontsize=14, fontweight="bold")
        plt.xlabel("Total Packets", fontsize=12)
        plt.ylabel("Protocol", fontsize=12)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{int(width):,}", (width, p.get_y() + p.get_height() / 2.),
                        ha="left", va="center", fontsize=10, xytext=(5, 0), textcoords="offset points")
        self._save_fig("investigation_anomalous_packets_by_protocol")

    def plot_investigation_top_source_ips(self, top_src: Dict[str, int]) -> None:
        """Investigation Plot 3: Top Anomalous Source IPs."""
        if not top_src:
            return
        plt.figure(figsize=(10, 6))
        series = pd.Series(top_src).head(10)
        ax = sns.barplot(x=series.values, y=series.index, hue=series.index, palette="YlOrRd_r", legend=False)
        plt.title("Top 10 Anomalous Source IP Addresses", fontsize=14, fontweight="bold")
        plt.xlabel("Anomalous Flow Count", fontsize=12)
        plt.ylabel("Source IP", fontsize=12)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{int(width):,}", (width, p.get_y() + p.get_height() / 2.),
                        ha="left", va="center", fontsize=10, xytext=(5, 0), textcoords="offset points")
        self._save_fig("investigation_top_source_ips")

    def plot_investigation_top_dest_ips(self, top_dst: Dict[str, int]) -> None:
        """Investigation Plot 4: Top Anomalous Destination IPs."""
        if not top_dst:
            return
        plt.figure(figsize=(10, 6))
        series = pd.Series(top_dst).head(10)
        ax = sns.barplot(x=series.values, y=series.index, hue=series.index, palette="Purples_r", legend=False)
        plt.title("Top 10 Anomalous Destination IP Addresses", fontsize=14, fontweight="bold")
        plt.xlabel("Anomalous Flow Count", fontsize=12)
        plt.ylabel("Destination IP", fontsize=12)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{int(width):,}", (width, p.get_y() + p.get_height() / 2.),
                        ha="left", va="center", fontsize=10, xytext=(5, 0), textcoords="offset points")
        self._save_fig("investigation_top_destination_ips")

    def plot_investigation_category_distribution(self, cat_dict: Dict[str, int]) -> None:
        """Investigation Plot 5: Potential Anomaly Category Breakdown."""
        if not cat_dict:
            return
        plt.figure(figsize=(10, 6))
        series = pd.Series(cat_dict)
        ax = sns.barplot(x=series.values, y=series.index, hue=series.index, palette="mako", legend=False)
        plt.title("Possible Anomaly Categories Breakdown", fontsize=14, fontweight="bold")
        plt.xlabel("Flow Count", fontsize=12)
        plt.ylabel("Possible Category", fontsize=12)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{int(width):,}", (width, p.get_y() + p.get_height() / 2.),
                        ha="left", va="center", fontsize=10, xytext=(5, 0), textcoords="offset points")
        self._save_fig("investigation_category_distribution")

    def plot_investigation_behaviour_distribution(self, beh_dict: Dict[str, int]) -> None:
        """Investigation Plot 6: Observable Behaviours Frequency."""
        if not beh_dict:
            return
        plt.figure(figsize=(10, 6))
        series = pd.Series(beh_dict).head(10)
        ax = sns.barplot(x=series.values, y=series.index, hue=series.index, palette="viridis", legend=False)
        plt.title("Most Common Observable Traffic Behaviours", fontsize=14, fontweight="bold")
        plt.xlabel("Trigger Count", fontsize=12)
        plt.ylabel("Behaviour Indicator", fontsize=12)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{int(width):,}", (width, p.get_y() + p.get_height() / 2.),
                        ha="left", va="center", fontsize=10, xytext=(5, 0), textcoords="offset points")
        self._save_fig("investigation_behaviour_distribution")

    def plot_investigation_model_agreement(self, anomaly_df: pd.DataFrame) -> None:
        """Investigation Plot 7: Model Consensus Agreement Matrix."""
        if anomaly_df.empty:
            return
        plt.figure(figsize=(8, 6))
        
        iso_anom = (anomaly_df["IsoForest_Pred"] == -1)
        ocsvm_anom = (anomaly_df["OneClassSVM_Pred"] == -1)
        db_anom = (anomaly_df["DBSCAN_Anomaly"] == -1)
        
        vote_count = iso_anom.astype(int) + ocsvm_anom.astype(int) + db_anom.astype(int)
        vote_dist = vote_count.value_counts().sort_index()

        labels = [
            f"0 Models (Normal: {vote_dist.get(0, 0):,})",
            f"1 Model (Minority: {vote_dist.get(1, 0):,})",
            f"2 Models (Consensus: {vote_dist.get(2, 0):,})",
            f"3 Models (Unanimous: {vote_dist.get(3, 0):,})",
        ]
        values = [vote_dist.get(0, 0), vote_dist.get(1, 0), vote_dist.get(2, 0), vote_dist.get(3, 0)]
        colors = ["#2ca02c", "#bcbd22", "#ff7f0e", "#d62728"]

        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)
        plt.title("ML Model Agreement & Consensus Anomaly Breakdown", fontsize=14, fontweight="bold")
        self._save_fig("investigation_model_agreement")

