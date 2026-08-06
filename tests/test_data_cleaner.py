"""
Unit Tests for DataCleaner Module.
"""

import pandas as pd
import pytest

from src.data_cleaner import DataCleaner


def test_data_cleaner_duplicates_and_invalid():
    """Tests duplicate removal and timestamp/length cleaning."""
    raw_data = pd.DataFrame({
        "Time": [0.0, 0.0, -1.0, 0.5, "invalid"],
        "Source": ["10.0.0.1", "10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.3"],
        "Destination": ["10.0.0.2", "10.0.0.2", "10.0.0.2", "10.0.0.1", "10.0.0.1"],
        "Protocol": ["tcp", "tcp", "tcp", "udp", None],
        "Length": [64, 64, 64, -50, 100],
        "Info": ["Info1", "Info1", "Info1", "Info2", "Info3"],
    })

    cleaner = DataCleaner(raw_data)
    cleaned_df, report = cleaner.clean_data()

    assert len(cleaned_df) == 1
    assert report["duplicate_rows_removed"] == 1
    assert report["malformed_rows_removed"] >= 1
    assert cleaned_df.iloc[0]["Protocol"] == "TCP"
