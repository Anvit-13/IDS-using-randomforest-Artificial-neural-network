# AI-Assisted Intrusion Detection System (IDS): Random Forest & Artificial Neural Network (ANN)

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)

An end-to-end, research-grade **Artificial Intelligence Intrusion Detection System (AI-IDS)**. This project transforms raw network packets and flow tables into **32 statistical flow features** to perform simultaneous, dual-model cyberattack classification using both a baseline **Random Forest Classifier** and a feedforward **Artificial Neural Network / Multilayer Perceptron (ANN / MLP)**.

---

## 📌 Project Architecture

```text
                     Input CSV / Live Network Traffic
                                     │
                                     ▼
                        5-Tuple Flow Construction
             (Source IP, Dest IP, Ports, Protocol, Timeout)
                                     │
                                     ▼
                           Exact 32 Flow Features
                                     │
                                     ▼
                         Shared RobustScaler Pipeline
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
         Random Forest (RF)                  ANN / MLP Neural Net
        [Tree-based Ensemble]                 [Backpropagation]
                    │                                 │
                    ▼                                 ▼
           Binary Detection                  Multiclass Threat Label
         (Normal vs. Attack)                (DDoS, Hulk, PortScan, etc.)
            Confidence: XX%                        Confidence: XX%
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                          Unified Decision Engine
                 ┌───────────────────────────────────────┐
                 │ Strategy: Consensus / Disagreement    │
                 │ Detection / Alert Resolution          │
                 └───────────────────────────────────────┘
                                     │
                                     ▼
                             Explanation Layer
                 (Key metrics: throughput, packet jitter, etc.)
                                     │
                                     ▼
                    Command-Line Alert & CSV Report
```

---

## 🚀 Key Highlights & Capabilities

- **Dual-Model Inference Engine**: Evaluates identical 32 flow features across two fundamentally different learning paradigms:
  - **Random Forest**: 200 decision trees with balanced class weights for fast, high-precision boundary decisions.
  - **ANN / MLP**: 32-input, 2-hidden layer (`32 -> 64 ReLU -> 32 ReLU -> 16 Softmax`) feedforward neural network for granular multiclass threat identification.
- **Identical 32-Feature Pipeline**: Strict feature alignment ensures fair evaluation across both models without feature leakage or drift.
- **Multi-Year Benchmark Training**: Trained and evaluated on combined, stratified datasets from both **CIC-IDS2017** and **CIC-IDS2018** (spanning 16+ million raw flows).
- **Consensus & Disagreement Strategy**: Automatically compares model predictions and flags **`MODEL DISAGREEMENT`** when the models differ, alerting security analysts to borderline traffic.
- **Evidence-Based Explanation Layer**: For every detected attack, extracts concrete contributing characteristics (e.g., packet rate spikes, IAT variance, payload variance) without hard-coded assumptions.
- **Flexible Modes**:
  - `--csv`: Process precomputed flow tables or raw Wireshark CSV packet captures.
  - `--live`: Real-time network interface packet sniffing and continuous flow detection.

---

## 📊 Benchmark Model Performance

Evaluated on held-out test traffic:

| Metric | Random Forest (Baseline) | Artificial Neural Network (ANN / MLP) |
| :--- | :---: | :---: |
| **Attack Detection Accuracy** | **94.40%** | **89.89%** |
| **Attack F1-Score** | **93.83%** | **88.10%** |
| **Multiclass Threat Categorization** | N/A (Binary) | **85.82%** |
| **Mutual Model Agreement** | **92.93%** | **92.93%** |
| **Inference Time (Batch)** | ~3.7 seconds | **~0.6 seconds** (6x faster) |

Supported attack classes (16 categories):
`BENIGN`, `DDoS` (LOIC, HOIC), `DoS Hulk`, `PortScan`, `DoS GoldenEye`, `Bot`, `FTP-Patator`, `SSH-Patator`, `DoS Slowhttptest`, `DoS slowloris`, `Web Attack - Brute Force`, `Web Attack - XSS`, `Web Attack - Sql Injection`, `Infiltration`, `Heartbleed`, and `Other Attack`.

---

## 📂 Project Structure

```text
├── data/
│   ├── cleaned/              # Cleaned benchmark parquet/csv files
│   ├── final/                # 32-feature dataset splits (train, val, test)
│   └── raw/                  # Raw packet captures
├── models/
│   ├── random_forest_baseline.pkl   # Serialized Random Forest model
│   ├── scaler.pkl                   # Shared RobustScaler pipeline
│   └── ann/
│       ├── ann_model.joblib         # Serialized ANN / MLP model
│       ├── label_encoder.joblib     # Label encoder for 16 classes
│       ├── class_mapping.json       # Class index mapping
│       └── ann_config.json          # ANN architecture hyperparameters
├── training/
│   ├── train_combined_models.py     # Joint 2017+2018 training script
│   └── train_ann.py                 # Standalone ANN training script
├── inference/
│   ├── rf_predictor.py              # Random Forest inference wrapper
│   ├── ann_predictor.py             # ANN / MLP inference wrapper
│   ├── unified_ids.py               # Dual-model orchestrator & disagreement engine
│   └── explainer.py                 # Statistical flow explanation layer
├── live/
│   ├── packet_capture.py            # Live network interface packet sniffer
│   └── flow_builder.py              # Real-time packet-to-32-feature flow assembler
├── reports/
│   └── flow_detection_report.csv    # Per-flow output report from CSV detection
├── src/                             # Legacy packet processing & GAN feature engineering
├── main.py                          # Unified CLI entrypoint
├── TEAM_PROJECT_REPORT.md           # Full technical report for teammates & reviewers
├── MODEL_COMPARISON.md              # Detailed benchmark metrics & comparison
├── PROJECT_ANALYSIS.md              # Initial technical analysis & feature definitions
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation
```

---

## ⚙️ Installation & Requirements

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/IDS-using-randomforest-Artificial-neural-network.git
cd IDS-using-randomforest-Artificial-neural-network

pip install -r requirements.txt
```

### 2. Optional: Live Capture Prerequisites
For running real-time live sniffing (`--live`):
1. Install Scapy: `pip install scapy`
2. On Windows: Install [Npcap](https://npcap.com/) (select *"Install Npcap in WinPcap API-compatible Mode"*).
3. Run terminal as Administrator.

---

## 🚀 Usage & CLI Examples

### 1. CSV Detection Mode (`--csv`)

Run dual RF + ANN detection on precomputed flow tables:
```bash
python main.py --csv "data/final/test.csv" --sample 5000
```

Run dual RF + ANN detection on raw Wireshark packet capture CSVs (automatically aggregates packets into 32-feature bidirectional flows):
```bash
python main.py --csv "NS CP PACKETS.csv" --sample 2000
```

#### Configurable Options:
- `--strategy`: Choose decision resolution strategy:
  - `consensus` (default): Report normal/attack when models agree; report `MODEL DISAGREEMENT` when they differ.
  - `high_recall`: Trigger alert if **either** model flags an attack.
  - `conservative`: Trigger alert only if **both** models flag an attack.
- `--output-report`: Custom path to export detailed per-flow CSV results (default: `reports/flow_detection_report.csv`).

Example:
```bash
python main.py --csv "NS CP PACKETS.csv" --strategy consensus --output-report "reports/audit_run.csv"
```

### 2. Live Packet Sniffing Mode (`--live`)

Monitor incoming and outgoing network traffic in real time:
```bash
python main.py --live
```
Or specify a specific interface:
```bash
python main.py --live --interface "Ethernet"
```

### 3. Retrain Combined Models

To retrain both Random Forest and ANN models on the combined CIC-IDS2017 and CIC-IDS2018 datasets:
```bash
python training/train_combined_models.py
```

### 4. Evaluate Models on Test Data

To run evaluation on held-out test splits and regenerate [MODEL_COMPARISON.md](file:///a:/college/SEM%205/AI/AI%20CP/MODEL_COMPARISON.md):
```bash
python evaluate_models.py
```

---

## 📋 Sample Output

```text
============================================================
AI-ASSISTED IDS REPORT
============================================================
Input: NS CP PACKETS.csv
Loaded 2,000 records in 0.06s.
Detected raw packet capture. Converting packets into 32-feature flows...
Generated 652 network flows.

Total flows: 652

Random Forest:
Normal: 652
Attack: 0

ANN:
Normal: 651
Attack: 1

Agreement: 99.8%
Model disagreement: 0.2%

Sample Flow Alert:
----------------------------------------
FLOW DETECTION
----------------------------------------
Source: 10.20.29.87
Destination: 172.192.176.118
Protocol: TCP

Random Forest:
Prediction: BENIGN
Score: 97.0%

ANN:
Prediction: DDoS
Score: 100.0%

Final status:
MODEL DISAGREEMENT

Predicted attack:
RF: BENIGN vs ANN: DDoS

Contributing characteristics:
 - Massive volumetric packet transmission rate (Flow Packets/s: 44.74)
 - Saturated bandwidth throughput (Flow Bytes/s: 39290.16)
 - High volume of unidirectional requests (Total Fwd Packets: 16.00)
----------------------------------------

Detailed per-flow report saved to: reports/flow_detection_report.csv
```

---

## 📚 Key Project Reports & References

- [TEAM_PROJECT_REPORT.md](file:///a:/college/SEM%205/AI/AI%20CP/TEAM_PROJECT_REPORT.md): Full comprehensive project writeup for teammates and course presentations.
- [MODEL_COMPARISON.md](file:///a:/college/SEM%205/AI/AI%20CP/MODEL_COMPARISON.md): Official metric comparison table and confusion matrices.
- [PROJECT_ANALYSIS.md](file:///a:/college/SEM%205/AI/AI%20CP/PROJECT_ANALYSIS.md): Feature dictionary, mathematical definitions, and preprocessing rules.
- [CIC_IDS2017_ANALYSIS_REPORT.md](file:///a:/college/SEM%205/AI/AI%20CP/CIC_IDS2017_ANALYSIS_REPORT.md): In-depth cleaning and statistical feature analysis of the CIC-IDS2017 corpus.
