"""
FlowInvestigator Module for Network Traffic Analysis & Intrusion Detection.

Provides dynamic creation of the Flow Investigation Table for consensus anomalous flows,
defensible rule-based behavioural tagging using population statistics, objective potential
category classification, flow-to-packet traceability, and representative flow selection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from src.logger import setup_logger

logger = setup_logger("FlowInvestigator")


class FlowInvestigator:
    """Class responsible for investigating consensus anomalous flows, tracing flows back

    to raw packet records, performing behavioural analysis, and assigning potential categories.
    """

    def __init__(
        self,
        flow_df: pd.DataFrame,
        anomaly_df: pd.DataFrame,
        packet_df: pd.DataFrame,
    ) -> None:
        """Initializes FlowInvestigator.

        Args:
            flow_df: Raw/unscaled flow DataFrame with 31 numeric features + metadata.
            anomaly_df: Anomaly prediction DataFrame containing ML outputs.
            packet_df: Cleaned packet DataFrame containing 'FlowID' column.
        """
        self.flow_df = flow_df.copy()
        self.anomaly_df = anomaly_df.copy()
        self.packet_df = packet_df.copy()

        # Merge flow features and metadata with ML anomaly predictions
        if "FlowID" in self.flow_df.columns and "FlowID" in self.anomaly_df.columns:
            cols_to_use = [c for c in self.anomaly_df.columns if c == "FlowID" or c not in self.flow_df.columns]
            self.merged_df = pd.merge(self.flow_df, self.anomaly_df[cols_to_use], on="FlowID", how="left")
        else:
            self.merged_df = self.flow_df.copy()

        # Compute dynamic statistical distribution thresholds across ALL flows
        self.thresholds: Dict[str, float] = self._compute_statistical_thresholds()

        # Build investigation dataset (consensus anomalous flows)
        self.investigation_df: pd.DataFrame = pd.DataFrame()
        self._build_investigation_dataset()

    def _compute_statistical_thresholds(self) -> Dict[str, float]:
        """Calculates statistically defensible percentiles across the full flow population."""
        pps = self.flow_df["Packets_Per_Second"].replace([np.inf, -np.inf], np.nan).dropna()
        bps = self.flow_df["Bytes_Per_Second"].replace([np.inf, -np.inf], np.nan).dropna()
        iat = self.flow_df["Mean_IAT"].replace([np.inf, -np.inf], np.nan).dropna()
        pkts = self.flow_df["Total_Packets"].dropna()
        bytes_col = self.flow_df["Total_Bytes"].dropna()

        thresholds = {
            "pps_95th": float(np.percentile(pps, 95)) if len(pps) > 0 else 1000.0,
            "bps_95th": float(np.percentile(bps, 95)) if len(bps) > 0 else 100000.0,
            "iat_5th": float(np.percentile(iat[iat > 0], 5)) if len(iat[iat > 0]) > 0 else 0.001,
            "pkts_95th": float(np.percentile(pkts, 95)) if len(pkts) > 0 else 38.0,
            "bytes_95th": float(np.percentile(bytes_col, 95)) if len(bytes_col) > 0 else 6947.0,
        }
        logger.info(f"Computed statistical distribution thresholds: {thresholds}")
        return thresholds

    def _build_investigation_dataset(self) -> pd.DataFrame:
        """Filters consensus anomalous flows (Consensus_Anomaly == -1),

        applies behavioural analysis, and assigns possible categories.
        """
        if "Consensus_Anomaly" not in self.merged_df.columns:
            logger.warning("Consensus_Anomaly column missing! Investigation DataFrame will be empty.")
            self.investigation_df = pd.DataFrame()
            return self.investigation_df

        anom_mask = self.merged_df["Consensus_Anomaly"] == -1
        self.investigation_df = self.merged_df[anom_mask].copy().reset_index(drop=True)

        logger.info(f"Extracted {len(self.investigation_df):,} consensus anomalous flows for investigation.")

        if self.investigation_df.empty:
            return self.investigation_df

        # Step 4: Behavioural Analysis
        behaviours_list = []
        for _, row in self.investigation_df.iterrows():
            behaviours_list.append(self._analyze_flow_behaviours(row))

        self.investigation_df["observed_behaviours"] = behaviours_list
        self.investigation_df["behaviours_str"] = [", ".join(b) for b in behaviours_list]

        # Step 5: Possible Category Assignment
        categories = []
        for _, row in self.investigation_df.iterrows():
            categories.append(self._assign_possible_category(row))

        self.investigation_df["possible_category"] = categories

        return self.investigation_df

    def _analyze_flow_behaviours(self, row: pd.Series) -> List[str]:
        """Evaluates statistical & protocol rules to return observable behavior tags."""
        behaviours = []

        proto = str(row.get("Protocol", "")).upper()
        pps = row.get("Packets_Per_Second", 0.0)
        bps = row.get("Bytes_Per_Second", 0.0)
        mean_iat = row.get("Mean_IAT", 0.0)
        pkts = row.get("Total_Packets", 0)
        total_bytes = row.get("Total_Bytes", 0)
        duration = row.get("Flow_Duration", 0.0)
        syn_cnt = row.get("SYN_Count", 0)
        ack_cnt = row.get("ACK_Count", 0)
        rst_cnt = row.get("RST_Count", 0)
        syn_ack_ratio = row.get("SYN_ACK_Ratio", 0.0)
        rst_ratio = row.get("RST_Ratio", 0.0)
        dst_ip = str(row.get("Destination_IP", ""))

        # 1. Packet Rate & Byte Rate
        if pps >= self.thresholds["pps_95th"] and pkts > 1 and duration > 0.001:
            behaviours.append("unusually_high_packet_rate")
        if bps >= self.thresholds["bps_95th"] or total_bytes >= self.thresholds["bytes_95th"]:
            behaviours.append("unusually_high_byte_rate")

        # 2. Inter-Arrival Time
        if pkts > 2 and 0.0 < mean_iat <= self.thresholds["iat_5th"]:
            behaviours.append("extremely_short_iat")

        # 3. TCP Specific Indicators
        if proto == "TCP":
            if syn_cnt > 0 and (syn_ack_ratio >= 2.0 or syn_cnt >= 5 or (syn_cnt > 0 and ack_cnt == 0)):
                behaviours.append("syn_heavy_tcp_behaviour")
            if rst_cnt >= 2 or rst_ratio >= 0.25:
                behaviours.append("excessive_rst_behaviour")

        # 4. Protocol Specific Volume & Patterns
        if proto == "ARP":
            if pkts >= 5 or pps > 10.0:
                behaviours.append("excessive_arp_activity")
        elif proto == "MDNS" or dst_ip in ["224.0.0.251", "ff02::fb"]:
            if pkts >= 10 or pps > 10.0:
                behaviours.append("excessive_mdns_multicast_activity")
        elif proto == "DNS":
            if pkts >= 5 or pps > 10.0 or total_bytes > 2000:
                behaviours.append("dns_query_bursts")
        elif proto in ["UDP", "QUIC"]:
            if total_bytes >= self.thresholds["bytes_95th"] or pkts >= self.thresholds["pkts_95th"]:
                behaviours.append("high_volume_udp_quic_traffic")
        elif proto in ["SSLV2", "SSL"]:
            behaviours.append("legacy_protocol_usage")

        # 5. Broadcast & Multicast
        if dst_ip in ["Broadcast", "255.255.255.255"] or dst_ip.endswith(".255"):
            behaviours.append("unusual_broadcast_traffic")
        elif (dst_ip.startswith("224.") or dst_ip.startswith("239.") or dst_ip.startswith("ff0")) and proto not in ["MDNS"]:
            behaviours.append("unusual_multicast_traffic")

        # Fallback if no specific rule triggered
        if not behaviours:
            behaviours.append("statistically_unusual_feature_profile")

        return behaviours

    def _assign_possible_category(self, row: pd.Series) -> str:
        """Assigns an objective possible category string based on observable behaviours and protocol attributes.

        Does NOT claim confirmed attack without ground truth labels.
        """
        behaviours = row.get("observed_behaviours", [])
        proto = str(row.get("Protocol", "")).upper()
        syn_cnt = row.get("SYN_Count", 0)
        ack_cnt = row.get("ACK_Count", 0)
        pkts = row.get("Total_Packets", 0)
        total_bytes = row.get("Total_Bytes", 0)

        # Priority rule mapping
        if "legacy_protocol_usage" in behaviours or proto in ["SSLV2", "SSL"]:
            return "Legacy protocol anomaly"

        if proto == "ARP" or "excessive_arp_activity" in behaviours:
            if "unusual_broadcast_traffic" in behaviours or pkts >= 20:
                return "Possible ARP spoofing/broadcast storm pattern"
            return "ARP anomaly"

        if "syn_heavy_tcp_behaviour" in behaviours:
            if syn_cnt > 0 and ack_cnt == 0 and pkts <= 5:
                return "Possible port scanning pattern"
            return "Possible SYN flood pattern"

        if proto == "DNS" or "dns_query_bursts" in behaviours:
            if total_bytes > 5000 or "unusually_high_byte_rate" in behaviours:
                return "Possible DNS tunneling pattern"
            return "DNS anomaly"

        if proto == "MDNS" or "excessive_mdns_multicast_activity" in behaviours:
            return "mDNS anomaly"

        if proto in ["UDP", "QUIC"] or "high_volume_udp_quic_traffic" in behaviours:
            return "Suspicious high-volume UDP/QUIC behaviour"

        if "excessive_rst_behaviour" in behaviours or (proto == "TCP" and ("unusually_high_packet_rate" in behaviours or "extremely_short_iat" in behaviours)):
            return "TCP anomaly"

        if "unusual_broadcast_traffic" in behaviours or "unusual_multicast_traffic" in behaviours:
            return "Unusual broadcast/multicast anomaly"

        if "unusually_high_packet_rate" in behaviours or "unusually_high_byte_rate" in behaviours:
            return "High-volume traffic anomaly"

        return "Unclassified anomaly"

    def get_packets_for_flow(self, flow_id: str) -> pd.DataFrame:
        """Retrieves original packet records belonging to a given FlowID.

        Args:
            flow_id: Canonical flow identifier string.

        Returns:
            pd.DataFrame: Original packet rows matching flow_id.
        """
        if "FlowID" not in self.packet_df.columns:
            logger.error("FlowID column missing from packet_df! Cannot trace packets.")
            return pd.DataFrame()

        matching_packets = self.packet_df[self.packet_df["FlowID"] == flow_id].copy()
        logger.debug(f"Retrieved {len(matching_packets)} packets for FlowID '{flow_id}'.")
        return matching_packets

    def get_packets_for_flows(self, flow_ids: List[str]) -> pd.DataFrame:
        """Retrieves original packet records for a list of FlowIDs.

        Args:
            flow_ids: List of canonical flow identifier strings.

        Returns:
            pd.DataFrame: Merged packet DataFrame for requested flow_ids.
        """
        if "FlowID" not in self.packet_df.columns:
            return pd.DataFrame()

        return self.packet_df[self.packet_df["FlowID"].isin(flow_ids)].copy()

    def generate_investigation_summary(self) -> Dict[str, Any]:
        """Generates comprehensive summary statistics from the investigation table.

        Returns:
            Dict[str, Any]: Summary dictionary containing counts, breakdowns, and top talkers.
        """
        if self.investigation_df.empty:
            return {}

        df = self.investigation_df
        total_anom = len(df)

        flows_by_proto = df["Protocol"].value_counts().to_dict()
        pkts_by_proto = df.groupby("Protocol")["Total_Packets"].sum().to_dict()
        bytes_by_proto = df.groupby("Protocol")["Total_Bytes"].sum().to_dict()

        top_src_ips = df["Source_IP"].value_counts().head(10).to_dict()
        top_dst_ips = df["Destination_IP"].value_counts().head(10).to_dict()

        top_src_ports = df[df["Source_Port"] > 0]["Source_Port"].value_counts().head(10).to_dict()
        top_dst_ports = df[df["Destination_Port"] > 0]["Destination_Port"].value_counts().head(10).to_dict()

        all_behaviours = [b for sublist in df["observed_behaviours"] for b in sublist]
        behaviour_counts = pd.Series(all_behaviours).value_counts().to_dict()
        category_counts = df["possible_category"].value_counts().to_dict()

        top_by_pkts = df.nlargest(5, "Total_Packets")[
            ["FlowID", "Protocol", "Source_IP", "Destination_IP", "Total_Packets"]
        ].to_dict(orient="records")
        top_by_bytes = df.nlargest(5, "Total_Bytes")[
            ["FlowID", "Protocol", "Source_IP", "Destination_IP", "Total_Bytes"]
        ].to_dict(orient="records")
        top_by_pps = df.nlargest(5, "Packets_Per_Second")[
            ["FlowID", "Protocol", "Source_IP", "Destination_IP", "Packets_Per_Second"]
        ].to_dict(orient="records")
        top_by_bps = df.nlargest(5, "Bytes_Per_Second")[
            ["FlowID", "Protocol", "Source_IP", "Destination_IP", "Bytes_Per_Second"]
        ].to_dict(orient="records")

        summary = {
            "total_consensus_anomalous_flows": total_anom,
            "anomalous_flows_by_protocol": flows_by_proto,
            "anomalous_packets_by_protocol": pkts_by_proto,
            "anomalous_bytes_by_protocol": bytes_by_proto,
            "top_anomalous_source_ips": top_src_ips,
            "top_anomalous_destination_ips": top_dst_ips,
            "top_anomalous_source_ports": top_src_ports,
            "top_anomalous_destination_ports": top_dst_ports,
            "most_common_behaviours": behaviour_counts,
            "most_common_possible_categories": category_counts,
            "top_flows_by_packet_count": top_by_pkts,
            "top_flows_by_byte_count": top_by_bytes,
            "top_flows_by_packet_rate": top_by_pps,
            "top_flows_by_byte_rate": top_by_bps,
        }
        return summary

    def get_representative_flows(self, category: str = "all", top_n: int = 20) -> pd.DataFrame:
        """Selects representative subsets of anomalous flows for deep forensic inspection.

        Args:
            category: Target criteria ('packet_count', 'byte_count', 'packet_rate',
                      'syn_heavy', 'arp', 'dns', 'mdns', 'udp_quic', 'sslv2', 'all').
            top_n: Number of flows to include per category selection.

        Returns:
            pd.DataFrame: Representative anomalous flows DataFrame.
        """
        df = self.investigation_df
        if df.empty:
            return pd.DataFrame()

        category = category.lower()
        if category == "packet_count":
            return df.nlargest(top_n, "Total_Packets")
        elif category == "byte_count":
            return df.nlargest(top_n, "Total_Bytes")
        elif category == "packet_rate":
            return df.nlargest(top_n, "Packets_Per_Second")
        elif category == "syn_heavy":
            tcp_df = df[df["Protocol"] == "TCP"]
            return tcp_df.nlargest(top_n, "SYN_Count")
        elif category == "arp":
            return df[df["Protocol"] == "ARP"].head(top_n)
        elif category == "dns":
            return df[df["Protocol"] == "DNS"].head(top_n)
        elif category == "mdns":
            return df[df["Protocol"] == "MDNS"].head(top_n)
        elif category == "udp_quic":
            return df[df["Protocol"].isin(["UDP", "QUIC"])].head(top_n)
        elif category == "sslv2":
            return df[df["Protocol"] == "SSLV2"].head(top_n)
        elif category == "all":
            rep_list = [
                df.nlargest(top_n, "Total_Packets"),
                df.nlargest(top_n, "Total_Bytes"),
                df.nlargest(top_n, "Packets_Per_Second"),
                df[df["Protocol"] == "TCP"].nlargest(top_n, "SYN_Count"),
                df[df["Protocol"] == "ARP"].head(top_n),
                df[df["Protocol"] == "DNS"].head(top_n),
                df[df["Protocol"] == "MDNS"].head(top_n),
                df[df["Protocol"].isin(["UDP", "QUIC"])].head(top_n),
                df[df["Protocol"] == "SSLV2"].head(top_n),
            ]
            combined = pd.concat(rep_list).drop_duplicates(subset=["FlowID"]).reset_index(drop=True)
            return combined
        else:
            return df.head(top_n)
