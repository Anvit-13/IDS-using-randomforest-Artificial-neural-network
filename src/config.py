"""
Configuration Manager Module.

Loads and validates YAML configuration settings for the pipeline.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from src.logger import setup_logger

logger = setup_logger("ConfigManager")


class ConfigManager:
    """Manages application configuration loaded from YAML file."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "pipeline": {
            "name": "IDS GAN Feature Extraction Pipeline",
            "random_seed": 42,
        },
        "data": {
            "input_path": "data/raw/NS CP PACKETS.csv",
            "processed_dir": "data/processed",
            "reports_dir": "data/reports",
            "plots_dir": "data/plots",
            "models_dir": "models",
            "logs_dir": "logs",
            "sampling_size": None,
        },
        "flow": {
            "flow_timeout": 120.0,
        },
        "visualization": {
            "dpi": 300,
            "style": "seaborn-v0_8-whitegrid",
            "palette": "deep",
        },
        "ml": {
            "isolation_forest": {
                "n_estimators": 100,
                "contamination": 0.05,
                "random_state": 42,
            },
            "one_class_svm": {
                "nu": 0.05,
                "kernel": "rbf",
                "gamma": "scale",
            },
            "dbscan": {
                "eps": 0.8,
                "min_samples": 5,
            },
            "kmeans": {
                "n_clusters": 4,
                "random_state": 42,
            },
        },
    }

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initializes configuration manager.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = Path(config_path) if config_path else Path("config/config.yaml")
        self.config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from YAML file, falling back to defaults if not found.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        if not self.config_path.exists():
            logger.warning(
                f"Configuration file '{self.config_path}' not found. Using default settings."
            )
            return self.DEFAULT_CONFIG.copy()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from '{self.config_path}'.")
                # Merge loaded config with defaults for missing keys
                return self._merge_dicts(self.DEFAULT_CONFIG, loaded_config or {})
        except Exception as e:
            logger.error(f"Error reading configuration file: {e}. Falling back to defaults.")
            return self.DEFAULT_CONFIG.copy()

    def _merge_dicts(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merges override dict into default dict."""
        merged = default.copy()
        for key, val in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
                merged[key] = self._merge_dicts(merged[key], val)
            else:
                merged[key] = val
        return merged

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieves a configuration value using dot notation (e.g. 'data.input_path').

        Args:
            key_path: Dot-separated key path.
            default: Default value if key is not found.

        Returns:
            Any: Value at key_path or default.
        """
        keys = key_path.split(".")
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val
