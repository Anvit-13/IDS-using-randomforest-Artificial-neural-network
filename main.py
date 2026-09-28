"""
Unified Main Entry Point for Intrusion Detection System (IDS).

Supports:
1. --csv <path>: Process an input CSV flow or Wireshark packet file through both
   Random Forest and ANN models, produce comparison metrics, and save detailed flow CSV reports.
2. --live: Real-time network interface packet capture, flow construction, and live attack alerts.
3. Legacy mode: -c/--config, -i/--input, -s/--sample runs the original GAN feature pipeline.
"""

from pathlib import Path
import argparse
import sys
import time
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

from inference.unified_ids import UnifiedIDS, FEATURE_NAMES
from live.packet_capture import LivePacketCapturer
from live.flow_builder import FlowBuilder
from src.cli import run_pipeline
from src.logger import setup_logger

logger = setup_logger("Main")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Assisted Unified Intrusion Detection System (Random Forest + ANN / MLP)."
    )
    # New Phase 6 & Phase 7 commands
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to an input CSV file (flows or raw packet capture) for dual RF + ANN detection.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Launch live network interface capture, flow construction, and dual-model IDS detection.",
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=None,
        help="Network interface name for live packet capture (default: system default).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="consensus",
        choices=["consensus", "high_recall", "conservative"],
        help="Combined model decision strategy: consensus, high_recall, or conservative.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/flow_detection_report.csv",
        help="Output CSV path for per-flow dual-model detection results.",
    )

    # Backward compatibility with existing pipeline flags
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file (legacy pipeline).",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        help="Path to input Wireshark CSV packet files (legacy pipeline).",
    )
    parser.add_argument(
        "-s",
        "--sample",
        type=int,
        help="Limit number of packets/rows to load for testing.",
    )
    return parser.parse_args()


def run_csv_mode(csv_path: str, strategy: str, output_report_path: str, sample_size: int = None) -> None:
    """
    Executes Phase 6: CSV Mode.
    Loads CSV, determines if it is a precomputed flow table or raw packets,
    applies the 32-feature extraction, scales identically, runs RF and ANN,
    outputs the AI-Assisted IDS Report, and saves a per-flow results CSV.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: Input CSV file '{csv_path}' does not exist.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("AI-ASSISTED IDS REPORT")
    print("=" * 60)
    print(f"Input: {csv_file.name}")

    # Load CSV
    t0 = time.time()
    df_raw = pd.read_csv(csv_file, nrows=sample_size, encoding="latin1")
    df_raw.columns = [c.strip() for c in df_raw.columns]
    load_time = time.time() - t0
    print(f"Loaded {len(df_raw):,} records in {load_time:.2f}s.")

    # Check if CSV is already in 32-feature flow format or raw Wireshark packets
    features_present = [f for f in FEATURE_NAMES if f in df_raw.columns]

    if len(features_present) == len(FEATURE_NAMES):
        print("Detected precomputed 32-feature flow table.")
        flow_df = df_raw.copy()
    else:
        # Check if raw packet capture
        if "Length" in df_raw.columns and "Time" in df_raw.columns:
            print("Detected raw packet capture. Converting packets into 32-feature flows...")
            builder = FlowBuilder()
            flow_df = builder.packets_to_flows(df_raw.to_dict("records"))
            print(f"Generated {len(flow_df):,} network flows.")
        else:
            missing = [f for f in FEATURE_NAMES if f not in df_raw.columns]
            print(f"Error: Provided CSV does not contain the required 32 features or valid packet fields.")
            print(f"Missing {len(missing)} features: {missing[:5]}...")
            sys.exit(1)

    total_flows = len(flow_df)
    if total_flows == 0:
        print("No valid flows found in input.")
        sys.exit(0)

    # Initialize Unified IDS and run predictions
    ids = UnifiedIDS(strategy=strategy)
    results_df = ids.predict_flows(flow_df)

    # Compute Summary Statistics
    rf_normal = int((results_df["RF_Prediction"] == "BENIGN").sum())
    rf_attack = int((results_df["RF_Prediction"] == "ATTACK").sum())

    ann_normal = int((results_df["ANN_Prediction"] == "BENIGN").sum())
    ann_attack = int((results_df["ANN_Prediction"] != "BENIGN").sum())

    agree_count = int((results_df["Model_Agreement"] == "AGREE").sum())
    disagree_count = total_flows - agree_count

    agreement_pct = (agree_count / total_flows) * 100.0
    disagreement_pct = (disagree_count / total_flows) * 100.0

    # Collect detected attack classes
    detected_attack_series = results_df[results_df["ANN_Prediction"] != "BENIGN"]["ANN_Prediction"].value_counts()

    # Print Summary Report
    print(f"\nTotal flows: {total_flows:,}")
    print("\nRandom Forest:")
    print(f"Normal: {rf_normal:,}")
    print(f"Attack: {rf_attack:,}")

    print("\nANN:")
    print(f"Normal: {ann_normal:,}")
    print(f"Attack: {ann_attack:,}")

    print(f"\nAgreement:\n{agreement_pct:.1f}%")
    print(f"\nModel disagreement:\n{disagreement_pct:.1f}%")

    print("\nAttack classes detected:")
    if detected_attack_series.empty:
        print("None (All traffic classified as Normal/Benign)")
    else:
        for attack_name, count in detected_attack_series.items():
            print(f"- {attack_name}: {count:,} flows")

    print("=" * 60)
    print("=" * 60)

    # Show sample individual flow alerts
    attack_indices = results_df[results_df["Final_Status"].isin(["ATTACK", "MODEL DISAGREEMENT"])].index
    if len(attack_indices) > 0:
        sample_idx = attack_indices[0]
        print("\nSample Flow Alert:")
        print(ids.format_single_report_str(results_df.iloc[sample_idx]) if hasattr(ids, "format_single_report_str") else ids.format_single_flow_report(results_df.iloc[sample_idx]))

    # Save detailed CSV report
    out_path = Path(output_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report_columns = [
        "FlowID", "Source_IP", "Destination_IP", "Protocol",
        "RF_Prediction", "RF_Score", "ANN_Prediction", "ANN_Score",
        "Final_Status", "Final_Predicted_Class", "Model_Agreement", "Explanation"
    ]
    # Keep only available columns
    save_cols = [c for c in report_columns if c in results_df.columns]
    results_df[save_cols].to_csv(out_path, index=False)
    print(f"\nDetailed per-flow report saved to: {out_path}")


def run_live_mode(interface: Optional[str], strategy: str) -> None:
    """
    Executes Phase 7: Live Detection Mode.
    Listens for live packets, groups them into flows, computes the 32 features,
    and displays live command-line detection alerts.
    """
    print("\n" + "=" * 60)
    print("LIVE AI INTRUSION DETECTION SYSTEM (RF + ANN)")
    print("=" * 60)

    supported, diag_msg = LivePacketCapturer.is_supported()
    if not supported:
        print(f"\n[!] Live Capture Interface Error:")
        print(f"    {diag_msg}")
        print("\nDocumentation & Missing Prerequisites for Live Sniffing:")
        print("1. Install Scapy: pip install scapy")
        print("2. On Windows: Install Npcap (https://npcap.com/) with 'WinPcap API-compatible mode' checked.")
        print("3. Run terminal / command prompt as Administrator.")
        print("\nNote: Fake/mock live packet data was explicitly NOT generated, per project constraints.")
        return

    print(f"Monitoring interface: {interface or 'default'}")
    print(f"Decision Strategy: {strategy}")
    print("Press Ctrl+C to terminate live detection.\n")

    capturer = LivePacketCapturer(interface=interface)
    builder = FlowBuilder()
    ids = UnifiedIDS(strategy=strategy)

    try:
        while True:
            # Capture small packet burst
            packet_records = capturer.capture_packets(packet_count=30, timeout=5)
            if not packet_records:
                print("[*] No packets received in capture window. Listening...")
                continue

            flows_df = builder.packets_to_flows(packet_records)
            if flows_df.empty:
                continue

            # Run dual inference
            results = ids.predict_flows(flows_df)

            # Output alerts
            for _, row in results.iterrows():
                print(ids.format_single_flow_report(row))
                print()

    except KeyboardInterrupt:
        print("\n[*] Live detection stopped by user.")


def main() -> None:
    """Main execution router."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_arguments()

    # Route 1: CSV Mode
    if args.csv:
        run_csv_mode(
            csv_path=args.csv,
            strategy=args.strategy,
            output_report_path=args.output_report,
            sample_size=args.sample
        )
        return

    # Route 2: Live Mode
    if args.live:
        run_live_mode(interface=args.interface, strategy=args.strategy)
        return

    # Route 3: Legacy Pipeline
    config_path = Path(args.config)
    input_paths = [Path(p) for p in args.input] if args.input else None
    sample_size = args.sample

    try:
        run_pipeline(
            config_path=config_path,
            input_paths=input_paths,
            sampling_size=sample_size,
        )
    except Exception as e:
        logger.critical(f"Pipeline execution failed with fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
