"""
Main Entry Point for Network Traffic Analysis and Feature Extraction Pipeline.

Allows CLI interaction, configuration overrides, multi-file merging, and execution of the full pipeline.
"""

from pathlib import Path
import argparse
import sys

from src.cli import run_pipeline
from src.logger import setup_logger

logger = setup_logger("Main")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Network Traffic Analysis and Feature Extraction Pipeline for GAN-Based Intrusion Detection Systems."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        help="Path to one or multiple input Wireshark CSV packet files (overrides config).",
    )
    parser.add_argument(
        "-s",
        "--sample",
        type=int,
        help="Limit number of packets to load for fast testing.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_arguments()

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
