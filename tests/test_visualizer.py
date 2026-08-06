"""
Unit Tests for Visualizer Module.
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.visualizer import Visualizer


def test_visualizer():
    """Tests saving plots to temporary directory."""
    packet_df = pd.DataFrame({
        "Time": [0.0, 1.0, 2.0],
        "Length": [64, 128, 256],
        "Protocol": ["TCP", "TCP", "UDP"],
        "IAT": [0.0, 1.0, 1.0],
    })

    flow_df = pd.DataFrame({
        "FlowID": ["F1", "F2"],
        "Flow_Duration": [1.0, 2.0],
        "Total_Bytes": [192, 256],
        "Protocol": ["TCP", "UDP"],
    })

    stat_results = {
        "top_source_ips": {"10.0.0.1": 2, "10.0.0.2": 1},
        "top_dest_ips": {"10.0.0.2": 2, "10.0.0.1": 1},
        "protocol_distribution": {"TCP": 2, "UDP": 1},
        "correlation_matrix": pd.DataFrame({"Duration": [1.0, 0.5], "Bytes": [0.5, 1.0]}),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        vis = Visualizer(output_dir=Path(temp_dir), dpi=100)
        vis.generate_all_plots(packet_df, flow_df, stat_results)

        png_files = list(Path(temp_dir).glob("*.png"))
        svg_files = list(Path(temp_dir).glob("*.svg"))

        assert len(png_files) >= 14
        assert len(svg_files) >= 14
