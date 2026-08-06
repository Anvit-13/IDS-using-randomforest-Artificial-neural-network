"""
Unit Tests for MLPipeline Module.
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.ml_pipeline import MLPipeline


def test_ml_pipeline():
    """Tests ML models fit, predict, scaling, and file saving."""
    flow_df = pd.DataFrame({
        "FlowID": [f"Flow_{i}" for i in range(20)],
        "Flow_Duration": [1.0 + i * 0.1 for i in range(20)],
        "Total_Packets": [i + 1 for i in range(20)],
        "Total_Bytes": [(i + 1) * 64 for i in range(20)],
        "Avg_Packet_Length": [64.0] * 20,
        "Packets_Per_Second": [10.0] * 20,
        "Bytes_Per_Second": [640.0] * 20,
        "Mean_IAT": [0.1] * 20,
        "Source_Port": [80] * 20,
        "Destination_Port": [54321] * 20,
    })

    with tempfile.TemporaryDirectory() as temp_dir:
        ml = MLPipeline(flow_df, models_dir=Path(temp_dir))
        gan_ready_df, anomaly_df = ml.run_pipeline()

        assert len(gan_ready_df) == 20
        assert len(anomaly_df) == 20
        assert "IsoForest_Pred" in anomaly_df.columns
        assert "Consensus_Anomaly" in anomaly_df.columns
        assert (Path(temp_dir) / "scaler.joblib").exists()
        assert (Path(temp_dir) / "isolation_forest.joblib").exists()
