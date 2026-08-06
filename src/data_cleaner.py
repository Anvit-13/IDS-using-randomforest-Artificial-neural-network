"""
DataCleaner Module for Wireshark Packet Captures.

Cleans packet capture DataFrames by removing duplicates, handling missing values,
normalizing types and protocol names, and dropping invalid records.
"""

from typing import Dict, Any, Tuple
import re
import pandas as pd
import numpy as np

from src.logger import setup_logger

logger = setup_logger("DataCleaner")


class DataCleaner:
    """Class responsible for cleaning and preprocessing Wireshark packet capture DataFrames."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initializes DataCleaner with a DataFrame.

        Args:
            df: Raw Wireshark packet DataFrame.
        """
        self.df = df.copy()
        self.cleaning_report: Dict[str, Any] = {
            "initial_row_count": len(df),
            "duplicate_rows_removed": 0,
            "malformed_rows_removed": 0,
            "missing_values_handled": 0,
            "impossible_values_removed": 0,
            "final_row_count": 0,
        }

    def clean_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Performs full cleaning pipeline on dataset.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and cleaning report.
        """
        logger.info("Starting Data Cleaning Process...")

        self._remove_duplicates()
        self._clean_timestamps()
        self._clean_packet_lengths()
        self._clean_protocols()
        self._clean_ip_addresses()
        self._handle_missing_values()
        self._remove_impossible_values()

        self.cleaning_report["final_row_count"] = len(self.df)
        logger.info(
            f"Data Cleaning Completed. Rows remaining: {self.cleaning_report['final_row_count']:,} "
            f"(Removed {self.cleaning_report['initial_row_count'] - self.cleaning_report['final_row_count']:,} invalid rows)."
        )

        return self.df, self.cleaning_report

    def _remove_duplicates(self) -> None:
        """Removes exact duplicate rows."""
        initial_len = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed = initial_len - len(self.df)
        self.cleaning_report["duplicate_rows_removed"] = removed
        if removed > 0:
            logger.info(f"Removed {removed:,} duplicate rows.")

    def _clean_timestamps(self) -> None:
        """Converts timestamp column to numeric float seconds."""
        if "Time" not in self.df.columns:
            return

        initial_len = len(self.df)
        self.df["Time"] = pd.to_numeric(self.df["Time"], errors="coerce")
        invalid_mask = self.df["Time"].isna() | (self.df["Time"] < 0)
        self.df = self.df[~invalid_mask].copy()

        # Sort chronologically by Time
        self.df.sort_values(by="Time", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        removed = initial_len - len(self.df)
        self.cleaning_report["malformed_rows_removed"] += removed
        if removed > 0:
            logger.info(f"Removed {removed:,} rows with invalid timestamp values.")

    def _clean_packet_lengths(self) -> None:
        """Converts Length column to integer and filters out non-positive or corrupted lengths."""
        if "Length" not in self.df.columns:
            return

        initial_len = len(self.df)
        self.df["Length"] = pd.to_numeric(self.df["Length"], errors="coerce")
        # Packet length must be > 0 (Wireshark frames are at least headers e.g. >= 14 or 20 bytes)
        valid_length_mask = (self.df["Length"] > 0) & (self.df["Length"] <= 65535)
        self.df = self.df[valid_length_mask].copy()
        self.df["Length"] = self.df["Length"].astype(int)

        removed = initial_len - len(self.df)
        self.cleaning_report["impossible_values_removed"] += removed
        if removed > 0:
            logger.info(f"Removed {removed:,} rows with invalid packet length (< 1 or > 65535).")

    def _clean_protocols(self) -> None:
        """Normalizes protocol strings (uppercase, stripped)."""
        if "Protocol" in self.df.columns:
            self.df["Protocol"] = (
                self.df["Protocol"]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN", "": "UNKNOWN"})
            )
        else:
            self.df["Protocol"] = "UNKNOWN"

    def _clean_ip_addresses(self) -> None:
        """Strips whitespace and normalizes Source and Destination columns."""
        for col in ["Source", "Destination"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
                # Fill missing or NaN strings with UNKNOWN
                self.df[col] = self.df[col].replace({"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})
            else:
                self.df[col] = "UNKNOWN"

    def _handle_missing_values(self) -> None:
        """Handles missing values across remaining columns."""
        if "Info" in self.df.columns:
            self.df["Info"] = self.df["Info"].fillna("").astype(str)

        null_count = self.df.isnull().sum().sum()
        if null_count > 0:
            self.df.dropna(subset=["Time", "Length"], inplace=True)
            self.df.fillna("UNKNOWN", inplace=True)
            self.cleaning_report["missing_values_handled"] = int(null_count)
            logger.info(f"Handled {null_count:,} missing values in non-critical columns.")

    def _remove_impossible_values(self) -> None:
        """Performs final sanity filter on packet attributes."""
        # Ensure non-negative frame numbers if present
        if "Frame_Number" in self.df.columns:
            self.df["Frame_Number"] = pd.to_numeric(self.df["Frame_Number"], errors="coerce").fillna(-1).astype(int)
