"""
Unit Tests for FlowInvestigator Module.
"""

import pandas as pd
import pytest

from src.flow_investigator import FlowInvestigator


def test_flow_investigator():
    """Tests FlowInvestigator dataset construction, behavioural tagging, category mapping, and traceability."""
    # Mock unscaled flow_df
    flow_df = pd.DataFrame({
        "FlowID": ["flow_1", "flow_2", "flow_3"],
        "Source_IP": ["10.0.0.1", "10.0.0.2", "192.168.1.5"],
        "Destination_IP": ["10.0.0.2", "224.0.0.251", "10.0.0.1"],
        "Source_Port": [54321, 5353, 80],
        "Destination_Port": [80, 5353, 54321],
        "Protocol": ["TCP", "MDNS", "TCP"],
        "Total_Packets": [100, 15, 2],
        "Total_Bytes": [50000, 1200, 128],
        "Flow_Duration": [2.0, 0.5, 0.0001],
        "Packets_Per_Second": [50.0, 30.0, 20000.0],
        "Bytes_Per_Second": [25000.0, 2400.0, 1280000.0],
        "Mean_IAT": [0.02, 0.03, 0.00005],
        "SYN_Count": [10, 0, 1],
        "ACK_Count": [2, 0, 0],
        "RST_Count": [0, 0, 0],
        "SYN_ACK_Ratio": [5.0, 0.0, 1.0],
        "RST_Ratio": [0.0, 0.0, 0.0],
    })

    # Mock ML anomaly_df
    anomaly_df = pd.DataFrame({
        "FlowID": ["flow_1", "flow_2", "flow_3"],
        "IsoForest_Pred": [-1, -1, 1],
        "OneClassSVM_Pred": [-1, 1, 1],
        "DBSCAN_Anomaly": [-1, -1, 1],
        "Consensus_Anomaly": [-1, -1, 1],  # flow_1 and flow_2 are consensus anomalies
    })

    # Mock packet_df
    packet_df = pd.DataFrame({
        "Frame_Number": [1, 2, 3, 4],
        "Time": [0.0, 0.1, 0.2, 0.3],
        "Source": ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.2"],
        "Destination": ["10.0.0.2", "10.0.0.2", "224.0.0.251", "224.0.0.251"],
        "Protocol": ["TCP", "TCP", "MDNS", "MDNS"],
        "Length": [64, 128, 80, 90],
        "FlowID": ["flow_1", "flow_1", "flow_2", "flow_2"],
    })

    investigator = FlowInvestigator(flow_df, anomaly_df, packet_df)
    inv_df = investigator.investigation_df

    # Step 2 Verification: Exactly 2 consensus anomalous flows included
    assert len(inv_df) == 2
    assert "flow_3" not in inv_df["FlowID"].values
    assert set(inv_df["FlowID"].values) == {"flow_1", "flow_2"}

    # Step 4 Verification: Behaviour tags assigned
    assert "observed_behaviours" in inv_df.columns
    flow_1_behaviours = inv_df[inv_df["FlowID"] == "flow_1"]["observed_behaviours"].values[0]
    assert "syn_heavy_tcp_behaviour" in flow_1_behaviours

    flow_2_behaviours = inv_df[inv_df["FlowID"] == "flow_2"]["observed_behaviours"].values[0]
    assert "excessive_mdns_multicast_activity" in flow_2_behaviours

    # Step 5 Verification: Possible Category assigned
    flow_1_cat = inv_df[inv_df["FlowID"] == "flow_1"]["possible_category"].values[0]
    assert flow_1_cat in ["Possible SYN flood pattern", "Possible port scanning pattern"]

    flow_2_cat = inv_df[inv_df["FlowID"] == "flow_2"]["possible_category"].values[0]
    assert flow_2_cat == "mDNS anomaly"

    # Step 3 Verification: Packet retrieval
    p1 = investigator.get_packets_for_flow("flow_1")
    assert len(p1) == 2
    assert (p1["FlowID"] == "flow_1").all()

    p2 = investigator.get_packets_for_flow("flow_2")
    assert len(p2) == 2

    # Step 6 Verification: Investigation summary
    summary = investigator.generate_investigation_summary()
    assert summary["total_consensus_anomalous_flows"] == 2
    assert summary["anomalous_flows_by_protocol"]["TCP"] == 1
    assert summary["anomalous_flows_by_protocol"]["MDNS"] == 1

    # Step 7 Verification: Representative flow selection
    rep_flows = investigator.get_representative_flows(category="all", top_n=5)
    assert len(rep_flows) >= 2
