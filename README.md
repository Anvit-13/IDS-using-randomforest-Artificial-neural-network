# Network Traffic Analysis and Feature Extraction Pipeline for GAN-Based Intrusion Detection Systems

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-PEP8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

An end-to-end, research-grade Python framework designed for undergraduate and graduate network security research. This project transforms raw Wireshark CSV packet captures into high-dimensional, clean, normalized, machine-learning-ready network flow features specifically engineered to train Generative Adversarial Networks (GANs) for Network Intrusion Detection Systems (NIDS).

> **Note**: The primary purpose of this project is **not** to detect attacks directly, but to extract, transform, clean, and standardize high-quality flow features that serve as training data for GAN Generator and Discriminator models in downstream intrusion detection research.

---

## 📌 Key Architectural Highlights

- **Modular & SOLID OOP Architecture**: Built with strict object-oriented design principles separating data loading, cleaning, feature extraction, flow aggregation, statistical analysis, visualization, ML modeling, and reporting.
- **Robust Wireshark CSV Ingestion**: Handles raw packet dumps with automatic encoding detection (`UTF-8`, `Latin-1`, `CP1252`), missing header aliasing, and port/flag parsing from raw Wireshark `Info` fields.
- **Bi-directional Flow Engine**: Reconstructs 5-tuple and 3-tuple network flows with configurable inactivity timeout splitting.
- **Statistical & Shannon Entropy Analysis**: Computes protocol frequencies, top talkers, Shannon entropy ($H(X) = -\sum p(x) \log_2 p(x)$), and higher statistical moments (mean, median, variance, skewness, kurtosis, 95% confidence intervals).
- **Publication-Quality Visualizations**: Automatically renders 14 distinct plot types exported in dual high-resolution **300 DPI PNG** and **SVG** vector formats.
- **Unsupervised Anomaly Detection Baseline**: Fits `IsolationForest`, `OneClassSVM`, `DBSCAN`, and `KMeans` models to evaluate feature separability and anomaly consensus scores.
- **Multi-Format Export Engine**: Exports structured reports in `CSV`, `JSON`, `TXT`, `Markdown`, and standalone styled `HTML`.

---

## 📂 Project Structure

```text
IDS_GAN_PIPELINE/
├── data/
│   ├── raw/                  # Input Wireshark packet capture CSVs
│   ├── processed/            # Cleaned packets and GAN-ready flow features CSVs
│   ├── reports/              # Summary metrics in CSV, JSON, TXT, MD, and HTML formats
│   └── plots/                # 14 Publication-grade 300 DPI PNG & SVG plots
├── models/                   # Serialized ML models and StandardScaler (.joblib)
├── logs/                     # Detailed execution logs (pipeline.log)
├── config/
│   └── config.yaml           # YAML configuration file
├── src/
│   ├── __init__.py
│   ├── logger.py             # Centralized logging module
│   ├── config.py             # ConfigManager with YAML fallback defaults
│   ├── data_loader.py        # DataLoader: CSV parsing, schema validation & stats
│   ├── data_cleaner.py       # DataCleaner: Duplicate removal & type normalization
│   ├── feature_extractor.py  # FeatureExtractor: Packet IAT, rolling statistics & flag regex
│   ├── flow_generator.py     # FlowGenerator: 5-tuple grouping & flow feature calculation
│   ├── stat_analyzer.py      # StatAnalyzer: Entropy, higher moments & correlation matrix
│   ├── visualizer.py         # Visualizer: 300 DPI publication plots (PNG + SVG)
│   ├── ml_pipeline.py        # MLPipeline: StandardScaler, Isolation Forest, OCSVM, DBSCAN, KMeans
│   ├── report_generator.py   # ReportGenerator: Exporting CSV, JSON, TXT, HTML & MD reports
│   └── cli.py                # Rich CLI interface and dataset merger
├── tests/                    # Comprehensive Pytest unit test suite
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_data_cleaner.py
│   ├── test_feature_extractor.py
│   ├── test_flow_generator.py
│   ├── test_stat_analyzer.py
│   ├── test_visualizer.py
│   └── test_ml_pipeline.py
├── main.py                   # CLI entrypoint
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git exclusion rules
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.12 or 3.13 installed.
- Git & Pip installed.

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-username/IDS_GAN_PIPELINE.git
cd IDS_GAN_PIPELINE

# Install required packages
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage

### 1. Standard Run (Default Config)
Place your Wireshark CSV capture (e.g., `NS CP PACKETS.csv`) into `data/raw/` and run:
```bash
python main.py
```

### 2. Custom Input & Config Override
Specify custom configuration files or directly target single/multiple packet captures:
```bash
# Run with explicit config file
python main.py --config config/config.yaml

# Run on a custom input CSV file
python main.py --input data/raw/NS\ CP\ PACKETS.csv

# Merge and process multiple CSV files simultaneously
python main.py --input file1.csv file2.csv file3.csv

# Fast testing with packet sampling (e.g., sample first 50,000 packets)
python main.py --sample 50000
```

---

## 🔄 18-Step Pipeline Workflow

1. **Step 1: Load Data**: `DataLoader` loads Wireshark CSVs, auto-detects delimiters and encodings, standardizes column aliases, and logs initial packet counts, memory usage, and missing values.
2. **Step 2: Data Cleaning**: `DataCleaner` removes duplicates, drops malformed or negative timestamps, normalizes protocol labels, and sanitizes IP fields.
3. **Step 3: Packet Feature Extraction**: `FeatureExtractor` computes Inter-Arrival Times (IAT), rolling average lengths, rolling std dev, instant packet rates, and size categories (Small/Medium/Large).
4. **Step 4: Flow Generation**: `FlowGenerator` groups packets into bidirectional 5-tuples (`Source IP`, `Destination IP`, `Source Port`, `Destination Port`, `Protocol`) or 3-tuples (`Source`, `Destination`, `Protocol`).
5. **Step 5: Flow Feature Calculation**: Computes duration, total packets, forward/backward counts, total bytes, min/max/mean/median/std/var length & IAT, active time, and idle time.
6. **Step 6: Advanced TCP Flag Processing**: Parses `SYN`, `ACK`, `FIN`, `RST`, `URG`, `PSH`, `ECE`, `CWR` counts and calculates ratios (`SYN/ACK`, `FIN/ACK`, `RST Ratio`).
7. **Step 7: Statistical Analysis**: `StatAnalyzer` calculates protocol distributions, top talkers, Shannon Entropy ($H(X)$), correlation matrices, and higher moments (mean, median, variance, skewness, kurtosis, 95% CI).
8. **Step 8: Publication-Quality Visualization**: `Visualizer` exports 14 plot types in 300 DPI PNG and SVG formats.
9. **Step 9: Machine Learning Baseline**: `MLPipeline` standardizes features with `StandardScaler` and fits `IsolationForest`, `OneClassSVM`, `DBSCAN`, and `KMeans`.
10. **Step 10: Multi-Format Reports**: `ReportGenerator` produces summary reports in CSV, JSON, TXT, MD, and HTML formats.
11. **Step 11: Output Data Persistence**: Exports standardized files:
    - `data/processed/clean_packets.csv`
    - `data/processed/flow_features.csv`
    - `data/reports/packet_statistics.csv`
    - `data/reports/protocol_statistics.csv`
    - `data/reports/anomaly_report.csv`
    - `data/reports/summary.json`
12. **Step 12: YAML Configuration**: Fully configurable parameters via `config/config.yaml`.
13. **Step 13: Centralized Logging**: Execution details logged to `logs/pipeline.log`.
14. **Step 14: Vectorized Performance**: High performance pandas/numpy vectorization supporting >1,000,000 packets.
15. **Step 15: Documentation**: Comprehensive README with research instructions.
16. **Step 16: Code Quality**: Clean modular OOP following PEP8 and SOLID principles.
17. **Step 17: Unit Testing**: Full Pytest coverage in `tests/`.
18. **Step 18: GAN Future Support**: Clean, non-null, normalized numerical feature outputs ready for PyTorch/TensorFlow GAN training.

---

## 📊 Visualizations Generated (`data/plots/`)

The pipeline automatically generates 14 publication-ready 300 DPI plot pairs (PNG & SVG):

| # | Figure Name | Description |
|---|---|---|
| 1 | `scatter_packet_length_vs_time` | Scatter plot of packet size over time colored by protocol |
| 2 | `scatter_flow_duration_vs_bytes` | Flow duration vs byte volume (log-scale) |
| 3 | `histogram_packet_size` | Histogram & KDE distribution of packet lengths |
| 4 | `histogram_flow_duration` | Histogram of flow durations |
| 5 | `histogram_inter_arrival_time` | Log-histogram of inter-arrival times |
| 6 | `boxplot_packet_length` | Packet length boxplots across top protocols |
| 7 | `boxplot_flow_duration` | Flow duration boxplots across top protocols |
| 8 | `barchart_protocol_distribution` | Bar chart of protocol frequencies with annotations |
| 9 | `barchart_top_source_ips` | Bar chart of top 10 Source IP talkers |
| 10 | `barchart_top_destination_ips` | Bar chart of top 10 Destination IP receivers |
| 11 | `heatmap_correlation_matrix` | Seaborn correlation heatmap of numerical flow features |
| 12 | `piechart_protocol_distribution` | Protocol share pie chart breakdown |
| 13 | `linegraph_packets_per_second` | Timeline graph of Packets Per Second (PPS) traffic intensity |
| 14 | `linegraph_bytes_per_second` | Timeline graph of Network Throughput (KB/s) |

---

## 🤖 Future GAN Integration Guide (Step 18)

The output dataset `data/processed/flow_features.csv` is preprocessed, numerical, standardized with `models/scaler.joblib`, and free of missing values.

### Loading into PyTorch for GAN Training Example:
```python
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Load GAN-ready feature dataset
df = pd.read_csv("data/processed/flow_features.csv")
features = df.drop(columns=["FlowID"]).values.astype("float32")

# Convert to PyTorch Tensor
dataset = TensorDataset(torch.tensor(features))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Generator Architecture Example
class FlowGeneratorGAN(torch.nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, feature_dim),
            torch.nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)
```

---

## 🧪 Running Unit Tests

Execute the unit test suite with Pytest:
```bash
pytest -v
```

---

## 📝 License & Citation

This project is open-source under the MIT License. Designed for undergraduate network security research in GAN-based Intrusion Detection Systems.
