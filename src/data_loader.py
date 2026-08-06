"""
DataLoader Module for Wireshark Packet Captures.

Handles loading, validating, and outputting initial dataset statistics for packet CSV files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import csv
import os
import pandas as pd

from src.logger import setup_logger

logger = setup_logger("DataLoader")


class DataLoader:
    """Class responsible for reading and validating Wireshark CSV packet captures."""

    REQUIRED_COLUMNS: List[str] = ["Time", "Length"]
    RECOMMENDED_COLUMNS: List[str] = ["Source", "Destination", "Protocol", "Info"]

    COLUMN_ALIAS_MAP: Dict[str, str] = {
        "No.": "Frame_Number",
        "No": "Frame_Number",
        "Time": "Time",
        "Source": "Source",
        "Src": "Source",
        "ip.src": "Source",
        "Destination": "Destination",
        "Dst": "Destination",
        "ip.dst": "Source",
        "Protocol": "Protocol",
        "Length": "Length",
        "len": "Length",
        "Info": "Info",
        "SrcPort": "Source_Port",
        "DstPort": "Destination_Port",
        "srcport": "Source_Port",
        "dstport": "Destination_Port",
    }

    def __init__(self, file_path: Union[str, Path]) -> None:
        """Initializes DataLoader with file path.

        Args:
            file_path: Path to the Wireshark CSV packet file.
        """
        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None
        self.stats: Dict[str, Union[int, float, Dict[str, int]]] = {}

    def auto_detect_delimiter(self) -> str:
        """Auto-detects delimiter (comma, semicolon, tab) of the CSV file.

        Returns:
            str: Detected delimiter string.
        """
        delimiter = ","
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(first_line)
                delimiter = dialect.delimiter
                logger.info(f"Auto-detected CSV delimiter: '{delimiter}'")
        except Exception as e:
            logger.warning(f"Failed to auto-detect delimiter: {e}. Defaulting to ','.")
        return delimiter

    def load_data(
        self,
        sampling_size: Optional[int] = None,
        delimiter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Reads CSV file into Pandas DataFrame with memory optimization and validation.

        Args:
            sampling_size: Optional maximum number of rows to load.
            delimiter: Optional explicit CSV delimiter.

        Returns:
            pd.DataFrame: Loaded and standardized dataframe.

        Raises:
            FileNotFoundError: If input file does not exist.
            ValueError: If file is empty or missing required columns.
        """
        logger.info(f"Loading data from '{self.file_path}'...")
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input data file '{self.file_path}' does not exist.")

        if self.file_path.stat().st_size == 0:
            raise ValueError(f"Input file '{self.file_path}' is empty.")

        sep = delimiter or self.auto_detect_delimiter()

        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        loaded_df = None
        last_exception = None

        for enc in encodings:
            try:
                if sampling_size:
                    logger.info(f"Sampling first {sampling_size:,} rows (encoding: {enc}).")
                    loaded_df = pd.read_csv(
                        self.file_path,
                        sep=sep,
                        nrows=sampling_size,
                        encoding=enc,
                        low_memory=False,
                        on_bad_lines="skip",
                    )
                else:
                    logger.info(f"Loading full dataset (encoding: {enc}).")
                    loaded_df = pd.read_csv(
                        self.file_path,
                        sep=sep,
                        encoding=enc,
                        low_memory=False,
                        on_bad_lines="skip",
                    )
                logger.info(f"Successfully read CSV with encoding '{enc}'.")
                break
            except (UnicodeDecodeError, Exception) as e:
                last_exception = e
                continue

        if loaded_df is None:
            logger.error(f"Error loading CSV file with all encodings: {last_exception}")
            raise last_exception

        self.df = loaded_df

        self._normalize_column_names()
        self.validate_schema()
        self._calculate_dataset_stats()

        logger.info(f"Successfully loaded {len(self.df):,} rows from dataset.")
        return self.df

    def _normalize_column_names(self) -> None:
        """Normalizes column names using standard aliases and strips whitespace."""
        if self.df is None:
            return

        # Strip whitespace and quotes
        cleaned_cols = {col: col.strip().strip('"') for col in self.df.columns}
        self.df.rename(columns=cleaned_cols, inplace=True)

        # Apply alias mapping where exact match exists
        rename_map = {}
        for col in self.df.columns:
            if col in self.COLUMN_ALIAS_MAP:
                rename_map[col] = self.COLUMN_ALIAS_MAP[col]

        if rename_map:
            self.df.rename(columns=rename_map, inplace=True)
            logger.info(f"Renamed columns for standardization: {rename_map}")

    def validate_schema(self) -> bool:
        """Validates presence of essential packet capture columns.

        Returns:
            bool: True if schema is valid.

        Raises:
            ValueError: If required columns are missing.
        """
        if self.df is None:
            raise ValueError("Data frame has not been loaded.")

        missing_required = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_required:
            raise ValueError(f"CSV dataset missing required columns: {missing_required}")

        missing_recommended = [col for col in self.RECOMMENDED_COLUMNS if col not in self.df.columns]
        if missing_recommended:
            logger.warning(f"CSV dataset missing recommended columns: {missing_recommended}")

        return True

    def _calculate_dataset_stats(self) -> Dict[str, Union[int, float, Dict[str, int]]]:
        """Calculates memory usage, packet count, missing value count."""
        if self.df is None:
            return {}

        num_packets = len(self.df)
        mem_usage_mb = float(self.df.memory_usage(deep=True).sum() / (1024 * 1024))
        missing_counts = self.df.isnull().sum().to_dict()

        self.stats = {
            "num_packets": num_packets,
            "memory_usage_mb": round(mem_usage_mb, 2),
            "columns_count": len(self.df.columns),
            "columns": list(self.df.columns),
            "missing_values": missing_counts,
        }
        return self.stats

    def display_statistics(self) -> None:
        """Prints dataset summary statistics to logger."""
        if not self.stats:
            self._calculate_dataset_stats()

        logger.info("=== Dataset Initial Statistics ===")
        logger.info(f"Total Packets Loaded: {self.stats['num_packets']:,}")
        logger.info(f"Memory Usage: {self.stats['memory_usage_mb']} MB")
        logger.info(f"Total Columns: {self.stats['columns_count']}")
        logger.info(f"Detected Columns: {self.stats['columns']}")
        logger.info(f"Missing Values: {self.stats['missing_values']}")
