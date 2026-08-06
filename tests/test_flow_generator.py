"""
Unit Tests for FlowGenerator Module.
"""

import pandas as pd
import pytest

from src.flow_generator import FlowGenerator


def test_flow_generator():
    """Tests grouping packets into flows and feature calculations."""
    packet_features_df = pd.DataFrame({
        "Time": [0.0, 0.1, 0.2, 200.0],
        "Source": ["10.0.0.1", "10.0.0.2", "10.0.0.1", "10.0.0.1"],
        "Destination": ["10.0.0.2", "10.0.0.1", "10.0.0.2", "10.0.0.2"],
        "Source_Port": [54321, 80, 54321, 54321],
        "Destination_Port": [80, 54321, 80, 80],
        "Protocol": ["TCP", "TCP", "TCP", "TCP"],
        "Length": [64, 64, 128, 64],
        "Flag_SYN": [1, 0, 0, 0],
        "Flag_ACK": [0, 1, 1, 0],
        "Flag_FIN": [0, 0, 0, 0],
        "Flag_RST": [0, 0, 0, 0],
        "Flag_URG": [0, 0, 0, 0],
        "Flag_PSH": [0, 0, 0, 0],
        "Flag_ECE": [0, 0, 0, 0],
        "Flag_CWR": [0, 0, 0, 0],
    })

    flow_gen = FlowGenerator(packet_features_df, flow_timeout=120.0)
    flow_df = flow_gen.generate_flows()

    assert len(flow_df) == 2  # The 4th packet arrived after 200s > 120s timeout -> split flow!
    assert "FlowID" in flow_df.columns
    assert "Total_Packets" in flow_df.columns
    assert flow_df.iloc[0]["Total_Packets"] == 3
    assert flow_df.iloc[0]["Forward_Packets"] == 2
    assert flow_df.iloc[0]["Backward_Packets"] == 1
