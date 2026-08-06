"""
FeatureExtractor Module for Wireshark Packet Captures.

Extracts packet-level features including Inter-Arrival Times (IAT), rolling statistical metrics,
packet size categorizations, port/flag parsing from Wireshark Info fields, and directional tagging.
"""

from typing import Tuple
import re
import numpy as np
import pandas as pd

from src.logger import setup_logger

logger = setup_logger("FeatureExtractor")


class FeatureExtractor:
    """Class responsible for extracting packet-level machine learning features."""

    # Common regex patterns for port extraction from Wireshark Info field
    PORT_PATTERN = re.compile(r"(\d{1,5})\s*(?:→|>|->)\s*(\d{1,5})")
    PORT_ALT_PATTERN = re.compile(r"SrcPort:\s*(\d{1,5}).*DstPort:\s*(\d{1,5})", re.IGNORECASE)

    # Standard Port mapping for protocol fallback
    DEFAULT_PROTOCOL_PORTS = {
        "HTTP": (80, 80),
        "HTTPS": (443, 443),
        "DNS": (53, 53),
        "MDNS": (5353, 5353),
        "SSDP": (1900, 1900),
        "DHCP": (67, 68),
        "NTP": (123, 123),
        "TLS": (443, 443),
        "TLSV1.2": (443, 443),
        "TLSV1.3": (443, 443),
        "SSL": (443, 443),
        "SSH": (22, 22),
        "FTP": (21, 21),
    }

    def __init__(self, df: pd.DataFrame, rolling_window: int = 10) -> None:
        """Initializes FeatureExtractor.

        Args:
            df: Cleaned packet DataFrame.
            rolling_window: Window size for rolling statistical metrics.
        """
        self.df = df.copy()
        self.rolling_window = rolling_window

    def extract_features(self) -> pd.DataFrame:
        """Runs the complete packet-level feature extraction pipeline.

        Returns:
            pd.DataFrame: Feature-enriched DataFrame.
        """
        logger.info("Extracting Packet-Level Features...")

        self._extract_iat()
        self._extract_rolling_metrics()
        self._extract_size_categories()
        self._extract_ports_and_flags()
        self._extract_direction()

        logger.info(f"Packet-level feature extraction completed. Features added: {len(self.df.columns)} columns.")
        return self.df

    def _extract_iat(self) -> None:
        """Calculates Inter-Arrival Time (IAT) between consecutive packets."""
        if "Time" in self.df.columns:
            # Shifted difference
            self.df["IAT"] = self.df["Time"].diff().fillna(0.0)
            self.df["IAT"] = np.maximum(0.0, self.df["IAT"])
        else:
            self.df["IAT"] = 0.0

    def _extract_rolling_metrics(self) -> None:
        """Calculates rolling and expanding statistical metrics over packet length and IAT."""
        window = self.rolling_window

        # Running (Expanding) Average Packet Length
        self.df["Running_Avg_Length"] = self.df["Length"].expanding(min_periods=1).mean()

        # Moving (Rolling) Average Packet Length
        self.df["Moving_Avg_Length"] = (
            self.df["Length"].rolling(window=window, min_periods=1).mean()
        )

        # Rolling Standard Deviation of Packet Length
        self.df["Rolling_Std_Length"] = (
            self.df["Length"].rolling(window=window, min_periods=1).std().fillna(0.0)
        )

        # Packet Rate (packets / sec in rolling window of time or count)
        # Avoid division by zero with clip
        safe_iat = self.df["IAT"].replace(0, np.nan).fillna(0.000001)
        self.df["Instant_Packet_Rate"] = 1.0 / safe_iat
        self.df["Instant_Bytes_Per_Sec"] = self.df["Length"] / safe_iat

        # Rolling average packet rate & bytes/sec over rolling window
        self.df["Rolling_Packet_Rate"] = (
            self.df["Instant_Packet_Rate"].rolling(window=window, min_periods=1).mean()
        )
        self.df["Rolling_Bytes_Per_Sec"] = (
            self.df["Instant_Bytes_Per_Sec"].rolling(window=window, min_periods=1).mean()
        )

    def _extract_size_categories(self) -> None:
        """Categorizes packets into Small, Medium, and Large sizes."""
        # Small: < 128 bytes, Medium: 128-512 bytes, Large: > 512 bytes
        bins = [-1, 127, 512, np.inf]
        labels = ["SMALL", "MEDIUM", "LARGE"]
        self.df["Packet_Size_Category"] = pd.cut(self.df["Length"], bins=bins, labels=labels)
        # One-hot encoding numeric columns for ML / GAN preparation
        self.df["Size_Small"] = (self.df["Packet_Size_Category"] == "SMALL").astype(int)
        self.df["Size_Medium"] = (self.df["Packet_Size_Category"] == "MEDIUM").astype(int)
        self.df["Size_Large"] = (self.df["Packet_Size_Category"] == "LARGE").astype(int)

    def _extract_ports_and_flags(self) -> None:
        """Extracts Source Port, Destination Port, and TCP flags from existing columns or Info string."""
        # Initialize default ports if not existing
        if "Source_Port" not in self.df.columns:
            self.df["Source_Port"] = -1
        else:
            self.df["Source_Port"] = pd.to_numeric(self.df["Source_Port"], errors="coerce").fillna(-1).astype(int)

        if "Destination_Port" not in self.df.columns:
            self.df["Destination_Port"] = -1
        else:
            self.df["Destination_Port"] = pd.to_numeric(self.df["Destination_Port"], errors="coerce").fillna(-1).astype(int)

        # Parse ports from Info column where port is missing (-1)
        if "Info" in self.df.columns:
            info_series = self.df["Info"].astype(str)

            # Vectorized regex extraction of ports
            extracted_ports = info_series.str.extract(r"(\d{1,5})\s*(?:→|>|->)\s*(\d{1,5})")
            mask_has_ports = extracted_ports[0].notna() & extracted_ports[1].notna()

            # Apply parsed ports where current port is -1
            src_ports_parsed = extracted_ports[0].fillna(-1).astype(int)
            dst_ports_parsed = extracted_ports[1].fillna(-1).astype(int)

            self.df["Source_Port"] = np.where(
                (self.df["Source_Port"] == -1) & mask_has_ports,
                src_ports_parsed,
                self.df["Source_Port"]
            )
            self.df["Destination_Port"] = np.where(
                (self.df["Destination_Port"] == -1) & mask_has_ports,
                dst_ports_parsed,
                self.df["Destination_Port"]
            )

            # Protocol fallback for remaining -1 ports
            for proto, (s_port, d_port) in self.DEFAULT_PROTOCOL_PORTS.items():
                proto_mask = self.df["Protocol"] == proto
                self.df.loc[proto_mask & (self.df["Source_Port"] == -1), "Source_Port"] = s_port
                self.df.loc[proto_mask & (self.df["Destination_Port"] == -1), "Destination_Port"] = d_port

            # Flag extraction from Info column
            self.df["Flag_SYN"] = info_series.str.contains(r"SYN", regex=True, case=False).astype(int)
            self.df["Flag_ACK"] = info_series.str.contains(r"ACK", regex=True, case=False).astype(int)
            self.df["Flag_FIN"] = info_series.str.contains(r"FIN", regex=True, case=False).astype(int)
            self.df["Flag_RST"] = info_series.str.contains(r"RST", regex=True, case=False).astype(int)
            self.df["Flag_URG"] = info_series.str.contains(r"URG", regex=True, case=False).astype(int)
            self.df["Flag_PSH"] = info_series.str.contains(r"PSH", regex=True, case=False).astype(int)
            self.df["Flag_ECE"] = info_series.str.contains(r"ECE", regex=True, case=False).astype(int)
            self.df["Flag_CWR"] = info_series.str.contains(r"CWR", regex=True, case=False).astype(int)
        else:
            for flag in ["SYN", "ACK", "FIN", "RST", "URG", "PSH", "ECE", "CWR"]:
                self.df[f"Flag_{flag}"] = 0

    def _extract_direction(self) -> None:
        """Infers direction (1 for Forward, 0 for Backward) based on IP hierarchy or first flow observation."""
        # Simple heuristic: If Source IP ends with lower octet or is private network initiator
        # Mark direction as Forward (1) vs Backward (0)
        if "Source" in self.df.columns and "Destination" in self.df.columns:
            # Sort order tag
            self.df["Direction"] = np.where(self.df["Source"] < self.df["Destination"], 1, 0)
        else:
            self.df["Direction"] = 1
