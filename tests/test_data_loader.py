"""
Unit Tests for DataLoader Module.
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.data_loader import DataLoader


@pytest.fixture
def sample_csv_path():
    """Creates a temporary sample Wireshark CSV file."""
    content = """"No.","Time","Source","Destination","Protocol","Length","Info"
"1","0.000000000","10.0.0.1","10.0.0.2","TCP","64","54321 → 80 [SYN] Seq=0 Win=65535 Len=0"
"2","0.001000000","10.0.0.2","10.0.0.1","TCP","64","80 → 54321 [SYN, ACK] Seq=0 Ack=1 Win=65535 Len=0"
"3","0.002000000","10.0.0.1","10.0.0.2","TCP","128","54321 → 80 [ACK] Seq=1 Ack=1 Win=65535 Len=64"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


def test_data_loader_valid(sample_csv_path):
    """Tests loading valid Wireshark CSV."""
    loader = DataLoader(sample_csv_path)
    df = loader.load_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "Time" in df.columns
    assert "Length" in df.columns
    assert "Source" in df.columns


def test_data_loader_missing_file():
    """Tests file not found exception handling."""
    loader = DataLoader(Path("non_existent_file.csv"))
    with pytest.raises(FileNotFoundError):
        loader.load_data()
