"""
Unit Tests for FeatureExtractor Module.
"""

import pandas as pd
import pytest

from src.feature_extractor import FeatureExtractor


def test_feature_extractor():
    """Tests IAT, size category, port, and flag extraction."""
    clean_df = pd.DataFrame({
        "Time": [0.0, 0.5, 1.5],
        "Source": ["10.0.0.1", "10.0.0.2", "10.0.0.1"],
        "Destination": ["10.0.0.2", "10.0.0.1", "10.0.0.2"],
        "Protocol": ["TCP", "TCP", "UDP"],
        "Length": [64, 300, 1000],
        "Info": [
            "54321 → 80 [SYN] Seq=0 Win=65535",
            "80 → 54321 [SYN, ACK] Seq=0 Ack=1",
            "1234 -> 53 len=1000",
        ],
    })

    extractor = FeatureExtractor(clean_df)
    df_feat = extractor.extract_features()

    assert "IAT" in df_feat.columns
    assert df_feat["IAT"].iloc[1] == 0.5
    assert "Flag_SYN" in df_feat.columns
    assert df_feat["Flag_SYN"].iloc[0] == 1
    assert df_feat["Flag_ACK"].iloc[1] == 1
    assert df_feat["Source_Port"].iloc[0] == 54321
    assert df_feat["Destination_Port"].iloc[0] == 80
