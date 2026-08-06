"""
Unit Tests for StatAnalyzer Module.
"""

import pandas as pd
import pytest

from src.stat_analyzer import StatAnalyzer


def test_stat_analyzer():
    """Tests summary statistics, Shannon entropy, and higher moments calculations."""
    packet_df = pd.DataFrame({
        "Time": [0.0, 1.0, 2.0],
        "Length": [64, 128, 256],
        "Source": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
        "Destination": ["10.0.0.2", "10.0.0.2", "10.0.0.1"],
        "Protocol": ["TCP", "TCP", "UDP"],
        "Source_Port": [80, 80, 53],
        "Destination_Port": [54321, 54321, 1234],
    })

    flow_df = pd.DataFrame({
        "FlowID": ["F1", "F2"],
        "Flow_Duration": [1.0, 2.0],
        "Total_Packets": [2, 1],
        "Total_Bytes": [192, 256],
        "Avg_Packet_Length": [96.0, 256.0],
        "Source_Port": [80, 53],
        "Destination_Port": [54321, 1234],
    })

    analyzer = StatAnalyzer(packet_df, flow_df)
    results = analyzer.run_full_analysis()

    assert "dataset_summary" in results
    assert results["dataset_summary"]["total_packets"] == 3
    assert "entropy" in results
    assert "entropy_source" in results["entropy"]
    assert results["entropy"]["entropy_source"] > 0
    assert "numerical_moments" in results
