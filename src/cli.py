"""
CLI Interface for IDS GAN Feature Extraction Pipeline.

Provides command-line argument parsing, multi-file merging, rich terminal rendering,
and orchestrates pipeline stage execution.
"""

from pathlib import Path
from typing import List, Optional
import argparse
import sys
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.config import ConfigManager
from src.data_cleaner import DataCleaner
from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.flow_generator import FlowGenerator
from src.logger import setup_logger
from src.ml_pipeline import MLPipeline
from src.report_generator import ReportGenerator
from src.stat_analyzer import StatAnalyzer
from src.visualizer import Visualizer

logger = setup_logger("CLI")
console = Console()


def merge_datasets(input_files: List[Path], sampling_size: Optional[int] = None) -> pd.DataFrame:
    """Loads and merges multiple Wireshark CSV packet capture files.

    Args:
        input_files: List of paths to input CSV files.
        sampling_size: Maximum packets per file if set.

    Returns:
        pd.DataFrame: Merged packet DataFrame.
    """
    dfs = []
    console.print(f"[bold cyan]Merging {len(input_files)} CSV files...[/bold cyan]")
    for idx, filepath in enumerate(input_files, 1):
        logger.info(f"Loading CSV file {idx}/{len(input_files)}: '{filepath}'")
        loader = DataLoader(filepath)
        df_part = loader.load_data(sampling_size=sampling_size)
        dfs.append(df_part)

    merged_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Successfully merged {len(input_files)} datasets. Total rows: {len(merged_df):,}")
    return merged_df


def run_pipeline(
    config_path: Path,
    input_paths: Optional[List[Path]] = None,
    sampling_size: Optional[int] = None,
) -> None:
    """Runs the complete 18-step IDS GAN Feature Extraction Pipeline.

    Args:
        config_path: Path to configuration YAML file.
        input_paths: Optional list of CSV input file paths.
        sampling_size: Optional maximum packet sample size.
    """
    console.print(
        Panel.fit(
            "[bold white on blue] IDS GAN Feature Extraction Pipeline [/bold white on blue]\n"
            "[italic gold1]Research-Grade Wireshark Packet Analysis & GAN Feature Engineering[/italic gold1]",
            border_style="blue",
        )
    )

    # 1. Load Configuration
    config = ConfigManager(config_path)
    sample_size = sampling_size or config.get("data.sampling_size")
    raw_input_path = input_paths or [Path(config.get("data.input_path"))]

    processed_dir = Path(config.get("data.processed_dir"))
    reports_dir = Path(config.get("data.reports_dir"))
    plots_dir = Path(config.get("data.plots_dir"))
    models_dir = Path(config.get("data.models_dir"))
    flow_timeout = float(config.get("flow.flow_timeout", 120.0))
    dpi = int(config.get("visualization.dpi", 300))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # Step 1: Load Data
        task_load = progress.add_task("[cyan]Step 1: Loading & Validating Data...", total=1)
        if len(raw_input_path) > 1:
            packet_df = merge_datasets(raw_input_path, sampling_size=sample_size)
        else:
            loader = DataLoader(raw_input_path[0])
            packet_df = loader.load_data(sampling_size=sample_size)
            loader.display_statistics()
        progress.update(task_load, completed=1)

        # Step 2: Data Cleaning
        task_clean = progress.add_task("[magenta]Step 2: Cleaning Data...", total=1)
        cleaner = DataCleaner(packet_df)
        cleaned_packet_df, cleaning_report = cleaner.clean_data()
        progress.update(task_clean, completed=1)

        # Step 3 & 6: Feature Extraction
        task_feat = progress.add_task("[green]Step 3 & 6: Extracting Packet-Level Features & TCP Flags...", total=1)
        extractor = FeatureExtractor(cleaned_packet_df)
        feature_packet_df = extractor.extract_features()
        progress.update(task_feat, completed=1)

        # Step 4 & 5: Flow Generation
        task_flow = progress.add_task("[yellow]Step 4 & 5: Generating Network Flows & Flow Metrics...", total=1)
        flow_gen = FlowGenerator(feature_packet_df, flow_timeout=flow_timeout)
        flow_df = flow_gen.generate_flows()
        progress.update(task_flow, completed=1)

        # Step 7: Statistical Analysis
        task_stat = progress.add_task("[blue]Step 7: Statistical Analysis & Shannon Entropy...", total=1)
        analyzer = StatAnalyzer(feature_packet_df, flow_df)
        stat_results = analyzer.run_full_analysis()
        progress.update(task_stat, completed=1)

        # Step 8: Visualization
        task_vis = progress.add_task("[red]Step 8: Generating 300 DPI Publication Plots...", total=1)
        visualizer = Visualizer(output_dir=plots_dir, dpi=dpi)
        visualizer.generate_all_plots(feature_packet_df, flow_df, stat_results)
        progress.update(task_vis, completed=1)

        # Step 9 & 18: Machine Learning & GAN Dataset Preparation
        task_ml = progress.add_task("[bright_cyan]Step 9 & 18: Machine Learning & GAN Preprocessing...", total=1)
        ml = MLPipeline(flow_df, models_dir=models_dir, config_params=config.get("ml"))
        gan_ready_df, anomaly_df = ml.run_pipeline()
        progress.update(task_ml, completed=1)

        # Step 10 & 11: Export Reports & Datasets
        task_rep = progress.add_task("[bright_green]Step 10 & 11: Exporting Reports & Processed Files...", total=1)
        reporter = ReportGenerator(reports_dir=reports_dir, processed_dir=processed_dir)
        reporter.export_processed_datasets(feature_packet_df, gan_ready_df)
        reporter.generate_all_reports(stat_results, cleaning_report, anomaly_df)
        progress.update(task_rep, completed=1)

    console.print(
        Panel.fit(
            "[bold green] PIPELINE EXECUTION SUCCESSFUL! [/bold green]\n\n"
            f"[white]Clean Packets File:[/white] {processed_dir / 'clean_packets.csv'}\n"
            f"[white]GAN Flow Features File:[/white] {processed_dir / 'flow_features.csv'}\n"
            f"[white]Anomaly Report:[/white] {reports_dir / 'anomaly_report.csv'}\n"
            f"[white]Summary Report:[/white] {reports_dir / 'summary.json'}\n"
            f"[white]Figures Saved to:[/white] {plots_dir} (14 PNG + SVG pairs, 300 DPI)\n"
            f"[white]Models Saved to:[/white] {models_dir}",
            border_style="green",
        )
    )
