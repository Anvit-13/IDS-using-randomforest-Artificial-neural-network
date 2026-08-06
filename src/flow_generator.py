"""
FlowGenerator Module for Wireshark Packet Captures.

Aggregates individual packets into bidirectional network flows (5-tuple / 3-tuple) with idle time splitting,
and computes comprehensive flow-level statistical, timing, and TCP flag features.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.logger import setup_logger

logger = setup_logger("FlowGenerator")


class FlowGenerator:
    """Class responsible for grouping packets into network flows and extracting flow-level features."""

    def __init__(self, df: pd.DataFrame, flow_timeout: float = 120.0) -> None:
        """Initializes FlowGenerator.

        Args:
            df: Feature-enriched packet DataFrame.
            flow_timeout: Timeout threshold in seconds to split inactive flows.
        """
        self.df = df.copy()
        self.flow_timeout = flow_timeout

    def generate_flows(self) -> pd.DataFrame:
        """Groups packets into flows and computes flow metrics.

        Returns:
            pd.DataFrame: DataFrame containing flow-level features.
        """
        logger.info(f"Generating Network Flows (Timeout: {self.flow_timeout}s)...")

        # 1. Establish Canonical Flow Keys (Bidirectional)
        self._assign_canonical_keys()

        # 2. Split Flows based on Flow Timeout
        self._assign_flow_ids()

        # 3. Compute Aggregated Flow Features
        flow_df = self._compute_flow_features()

        logger.info(f"Flow generation completed. Generated {len(flow_df):,} total network flows.")
        return flow_df

    def _assign_canonical_keys(self) -> None:
        """Creates canonical bidirectional keys for 5-tuple / 3-tuple flow grouping."""
        src_ip = self.df["Source"].values
        dst_ip = self.df["Destination"].values
        src_port = self.df["Source_Port"].values
        dst_port = self.df["Destination_Port"].values
        proto = self.df["Protocol"].values

        # Determine canonical order so (Src, Dst) and (Dst, Src) map to same key
        is_src_smaller = src_ip <= dst_ip

        ip1 = np.where(is_src_smaller, src_ip, dst_ip)
        ip2 = np.where(is_src_smaller, dst_ip, src_ip)
        port1 = np.where(is_src_smaller, src_port, dst_port)
        port2 = np.where(is_src_smaller, dst_port, src_port)

        self.df["Canonical_IP1"] = ip1
        self.df["Canonical_IP2"] = ip2
        self.df["Canonical_Port1"] = port1
        self.df["Canonical_Port2"] = port2

        # Record direction relative to canonical IP1: 1 if Forward (Src == IP1), 0 if Backward
        self.df["Is_Forward"] = (src_ip == ip1).astype(int)

    def _assign_flow_ids(self) -> None:
        """Assigns unique FlowIDs based on 5-tuple and inactivity timeout."""
        # Ensure chronological order
        self.df.sort_values(by=["Canonical_IP1", "Canonical_IP2", "Canonical_Port1", "Canonical_Port2", "Protocol", "Time"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Compute per-flow packet IAT
        flow_group_cols = ["Canonical_IP1", "Canonical_IP2", "Canonical_Port1", "Canonical_Port2", "Protocol"]
        group_time_diff = self.df.groupby(flow_group_cols)["Time"].diff().fillna(0.0)

        # Increment sub-flow ID whenever IAT > flow_timeout
        self.df["Is_New_Subflow"] = (group_time_diff > self.flow_timeout).astype(int)
        self.df["Subflow_ID"] = self.df.groupby(flow_group_cols)["Is_New_Subflow"].cumsum()

        # Combine tuple + subflow index into string FlowID
        self.df["FlowID"] = (
            self.df["Canonical_IP1"].astype(str) + "_" +
            self.df["Canonical_IP2"].astype(str) + "_" +
            self.df["Canonical_Port1"].astype(str) + "_" +
            self.df["Canonical_Port2"].astype(str) + "_" +
            self.df["Protocol"].astype(str) + "_F" +
            self.df["Subflow_ID"].astype(str)
        )

    def _compute_flow_features(self) -> pd.DataFrame:
        """Computes all required flow metrics using groupby aggregations."""
        # Compute IAT within each flow group
        self.df["Flow_IAT"] = self.df.groupby("FlowID")["Time"].diff().fillna(0.0)

        # Base Aggregations
        grouped = self.df.groupby("FlowID")

        # Aggregation Dictionary
        agg_dict = {
            "Source": "first",
            "Destination": "first",
            "Source_Port": "first",
            "Destination_Port": "first",
            "Protocol": "first",
            "Time": ["min", "max"],
            "Length": ["count", "sum", "mean", "median", "min", "max", "var", "std"],
            "Flow_IAT": ["mean", "median", "min", "max", "std"],
            "Is_Forward": ["sum"],
            "Flag_SYN": "sum",
            "Flag_ACK": "sum",
            "Flag_FIN": "sum",
            "Flag_RST": "sum",
            "Flag_URG": "sum",
            "Flag_PSH": "sum",
            "Flag_ECE": "sum",
            "Flag_CWR": "sum",
        }

        agg_df = grouped.agg(agg_dict)

        # Flatten multi-level columns
        agg_df.columns = [
            "Source_IP", "Destination_IP", "Source_Port", "Destination_Port", "Protocol",
            "Start_Time", "End_Time",
            "Total_Packets", "Total_Bytes", "Avg_Packet_Length", "Median_Packet_Length",
            "Min_Packet_Length", "Max_Packet_Length", "Packet_Length_Var", "Packet_Length_Std",
            "Mean_IAT", "Median_IAT", "Min_IAT", "Max_IAT", "Std_IAT",
            "Forward_Packets",
            "SYN_Count", "ACK_Count", "FIN_Count", "RST_Count", "URG_Count", "PSH_Count", "ECE_Count", "CWR_Count"
        ]

        agg_df.reset_index(inplace=True)

        # Fill NaNs in std/var for 1-packet flows
        agg_df["Packet_Length_Var"] = agg_df["Packet_Length_Var"].fillna(0.0)
        agg_df["Packet_Length_Std"] = agg_df["Packet_Length_Std"].fillna(0.0)
        agg_df["Std_IAT"] = agg_df["Std_IAT"].fillna(0.0)

        # Derived Flow Features
        agg_df["Backward_Packets"] = agg_df["Total_Packets"] - agg_df["Forward_Packets"]
        agg_df["Flow_Duration"] = np.maximum(0.000001, agg_df["End_Time"] - agg_df["Start_Time"])

        # Rates
        agg_df["Packets_Per_Second"] = agg_df["Total_Packets"] / agg_df["Flow_Duration"]
        agg_df["Bytes_Per_Second"] = agg_df["Total_Bytes"] / agg_df["Flow_Duration"]

        # Idle & Active Time calculation
        # Idle time is defined as sum of IATs >= 1.0s within the flow
        idle_threshold = 1.0
        idle_df = self.df[self.df["Flow_IAT"] >= idle_threshold].groupby("FlowID")["Flow_IAT"].sum().reset_index()
        idle_df.columns = ["FlowID", "Flow_Idle_Time"]

        agg_df = pd.merge(agg_df, idle_df, on="FlowID", how="left")
        agg_df["Flow_Idle_Time"] = agg_df["Flow_Idle_Time"].fillna(0.0)
        agg_df["Flow_Active_Time"] = np.maximum(0.0, agg_df["Flow_Duration"] - agg_df["Flow_Idle_Time"])

        # Flag Ratios
        eps = 1e-6
        agg_df["SYN_ACK_Ratio"] = agg_df["SYN_Count"] / (agg_df["ACK_Count"] + eps)
        agg_df["FIN_ACK_Ratio"] = agg_df["FIN_Count"] / (agg_df["ACK_Count"] + eps)
        agg_df["RST_Ratio"] = agg_df["RST_Count"] / (agg_df["Total_Packets"] + eps)

        # Reorder columns logically
        columns_order = [
            "FlowID", "Source_IP", "Destination_IP", "Source_Port", "Destination_Port", "Protocol",
            "Flow_Duration", "Total_Packets", "Forward_Packets", "Backward_Packets", "Total_Bytes",
            "Avg_Packet_Length", "Median_Packet_Length", "Min_Packet_Length", "Max_Packet_Length",
            "Packet_Length_Var", "Packet_Length_Std", "Packets_Per_Second", "Bytes_Per_Second",
            "Mean_IAT", "Median_IAT", "Min_IAT", "Max_IAT", "Std_IAT",
            "Flow_Idle_Time", "Flow_Active_Time",
            "SYN_Count", "ACK_Count", "FIN_Count", "RST_Count", "URG_Count", "PSH_Count", "ECE_Count", "CWR_Count",
            "SYN_ACK_Ratio", "FIN_ACK_Ratio", "RST_Ratio"
        ]

        flow_features_df = agg_df[columns_order].copy()
        return flow_features_df
