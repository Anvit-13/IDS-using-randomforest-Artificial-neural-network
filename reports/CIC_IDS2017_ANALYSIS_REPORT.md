# CIC-IDS2017 Dataset Preparation and Feature Analysis Report

## Executive Summary
This report presents a thorough, reproducible preparation, cleaning, and statistical feature analysis of the complete **CIC-IDS2017** network flow dataset. The dataset comprises network traffic captured over five consecutive days, covering **8 individual CSV files**, **2,830,743 raw network flow records**, **78 numerical traffic features**, and **15 original traffic/attack labels**.

> [!IMPORTANT]
> **Data Integrity Guarantee:** No features have been deleted or finalized into a 32-feature subset. All 78 valid features are preserved in the cleaned dataset (`data/cleaned/cleaned_cic_ids2017.parquet` and `data/cleaned/cleaned_cic_ids2017.csv`) to enable downstream AI and domain-expert evaluation.

---

## 1. Dataset Locating & File-by-File Statistics

The analysis processed **8 raw CSV files** located in the workspace under `cic ids 2017/MachineLearningCVE`:

| File Name | Size (MB) | Raw Rows | Raw Columns | Unique Traffic Labels |
| :--- | ---: | ---: | ---: | :--- |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 73.55 | 225,745 | 79 | DDoS (128,027), BENIGN (97,718)... |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 73.34 | 286,467 | 79 | PortScan (158,930), BENIGN (127,537)... |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 55.62 | 191,033 | 79 | BENIGN (189,067), Bot (1,966)... |
| `Monday-WorkingHours.pcap_ISCX.csv` | 168.73 | 529,918 | 79 | BENIGN (529,918)... |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 79.25 | 288,602 | 79 | BENIGN (288,566), Infiltration (36)... |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 49.61 | 170,366 | 79 | BENIGN (168,186), Web Attack ï¿½ Brute Force (1,507), Web Attack ï¿½ XSS (652)... |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 128.82 | 445,909 | 79 | BENIGN (432,074), FTP-Patator (7,938), SSH-Patator (5,897)... |
| `Wednesday-workingHours.pcap_ISCX.csv` | 214.74 | 692,703 | 79 | BENIGN (440,031), DoS Hulk (231,073), DoS GoldenEye (10,293)... |

---

## 2. Combined Dataset Overview & Column Normalization

* **Total Raw Rows:** `2,830,743`
* **Total Raw Columns:** `79`
* **Total Traffic Features:** `78`
* **Target Label Column Name:** `Label` (normalized from `' Label'`)
* **Column Compatibility Across Files:** `100% Match` (All 8 files contain identical 79 raw headers in identical sequence).
* **Header Normalization:** Leading and trailing whitespace were stripped across all column names. Explicit mapping was created between original raw headers and cleaned feature names (e.g. `' Destination Port'` $\rightarrow$ `'Destination Port'`). Duplicate column name `' Fwd Header Length.1'` was explicitly indexed.

---

## 3. Data Cleaning & Conservative Filtering Audit

A conservative cleaning strategy was executed without destroying legitimate attack spikes:

### A. Missing Values
* **Total Missing Values Detected:** `1,358`
* **Affected Features:** Only `Flow Bytes/s` contained `1,358` missing entries (0.048% of total raw rows).
* **Treatment:** Imputed using the feature median in the cleaned dataset.

### B. Infinite Values
* **Total Infinite Values Detected:** `4,376`
* **Affected Features:** `Flow Bytes/s` (`1,509` Inf values) and `Flow Packets/s` (`2,867` Inf values).
* **Cause:** Division by zero when `Flow Duration = 0`.
* **Treatment:** Replaced `+Inf`/`-Inf` with feature medians in the cleaned dataset.

### C. Exact Duplicate Rows
* **Exact Duplicate Rows Identified:** `308,381` (`10.89%` of total raw dataset).
* **Treatment:** Exact duplicate rows were removed from the cleaned dataset.
* **Rows After Deduplication:** `2,522,362`.

### D. Invalid Data vs. Legitimate Extreme Network Behavior
* **Header Length Integer Overflow:** Features `Fwd Header Length`, `Bwd Header Length`, `Fwd Header Length.1`, and `min_seg_size_forward` exhibited 32-bit signed integer overflow negative values (e.g. `-32,212,234,632`) due to CICFlowMeter counter rollover during high-volume DDoS flooding. Overflow values were corrected to feature medians.
* **Negative Flow Durations / IATs:** 115 records showed negative flow durations (min `-13` $\mu s$) due to packet capture timestamp jitter; set to `0`.
* **TCP Initial Window Special Values (`-1`):** `Init_Win_bytes_forward` (1,001,189 rows with `-1`) and `Init_Win_bytes_backward` (1,441,552 rows with `-1`) contain `-1` to denote non-TCP traffic (UDP/ICMP) or uncaptured SYN packets. **These values were intentionally preserved as valid domain indicators.**

---

## 4. Traffic Label & Attack Family Distribution

The cleaned dataset contains **15 distinct original attack/traffic labels**, grouped into **9 high-level attack families**:

### Original Label Distribution

| Label | Cleaned Samples | Percentage (%) | Attack Family Group |
| :--- | ---: | ---: | :--- |
| `BENIGN` | 2,096,484 | 83.1159% | BENIGN |
| `DoS Hulk` | 172,849 | 6.8527% | DoS |
| `DDoS` | 128,016 | 5.0752% | DoS |
| `PortScan` | 90,819 | 3.6006% | Port Scan |
| `DoS GoldenEye` | 10,286 | 0.4078% | DoS |
| `FTP-Patator` | 5,933 | 0.2352% | Brute Force |
| `DoS slowloris` | 5,385 | 0.2135% | DoS |
| `DoS Slowhttptest` | 5,228 | 0.2073% | DoS |
| `SSH-Patator` | 3,219 | 0.1276% | Brute Force |
| `Bot` | 1,953 | 0.0774% | Botnet |
| `Web Attack ï¿½ Brute Force` | 1,470 | 0.0583% | Brute Force |
| `Web Attack ï¿½ XSS` | 652 | 0.0258% | Web Attack |
| `Infiltration` | 36 | 0.0014% | Infiltration |
| `Web Attack ï¿½ Sql Injection` | 21 | 0.0008% | Web Attack |
| `Heartbleed` | 11 | 0.0004% | Heartbleed |

### Attack Family Summary

| Attack Family | Combined Samples | Percentage (%) | Key Attack Subtypes |
| :--- | ---: | ---: | :--- |
| `BENIGN` | 2,096,484 | 83.1159% | Standardized Family Category |
| `DoS` | 321,764 | 12.7565% | Standardized Family Category |
| `Port Scan` | 90,819 | 3.6006% | Standardized Family Category |
| `Brute Force` | 10,622 | 0.4211% | Standardized Family Category |
| `Botnet` | 1,953 | 0.0774% | Standardized Family Category |
| `Web Attack` | 673 | 0.0267% | Standardized Family Category |
| `Infiltration` | 36 | 0.0014% | Standardized Family Category |
| `Heartbleed` | 11 | 0.0004% | Standardized Family Category |

> [!NOTE]
> **Rare Attack Warning:** `Heartbleed` (11 samples, 0.0004%) and `Infiltration` (36 samples, 0.0014%) contain very small sample sizes. Statistical conclusions for these two categories should be treated with caution.

---

## 5. Feature Redundancy & Correlation Analysis

Correlation analysis identified **102 highly correlated feature pairs** ($|r| \ge 0.90$) and **68 extremely correlated pairs** ($|r| \ge 0.95$).

### Top Extremely Correlated & Duplicate Feature Groups ($|r| \ge 0.95$)

| Feature A | Feature B | Correlation ($r$) | Redundancy Diagnosis |
| :--- | :--- | ---: | :--- |
| `Total Fwd Packets` | `Total Backward Packets` | `0.999048` | Perfectly redundant mathematical duplicate |
| `Total Fwd Packets` | `Total Length of Bwd Packets` | `0.995845` | Perfectly redundant mathematical duplicate |
| `Total Backward Packets` | `Total Length of Bwd Packets` | `0.997255` | Perfectly redundant mathematical duplicate |
| `Fwd Packet Length Max` | `Fwd Packet Length Std` | `0.970153` | Perfectly redundant mathematical duplicate |
| `Bwd Packet Length Max` | `Bwd Packet Length Mean` | `0.958355` | Perfectly redundant mathematical duplicate |
| `Bwd Packet Length Max` | `Bwd Packet Length Std` | `0.982465` | Perfectly redundant mathematical duplicate |
| `Flow Duration` | `Fwd IAT Total` | `0.998670` | Perfectly redundant mathematical duplicate |
| `Flow IAT Max` | `Fwd IAT Max` | `0.998252` | Perfectly redundant mathematical duplicate |
| `Total Fwd Packets` | `Fwd Header Length` | `0.998479` | Perfectly redundant mathematical duplicate |
| `Total Backward Packets` | `Fwd Header Length` | `0.996664` | Perfectly redundant mathematical duplicate |
| `Total Length of Bwd Packets` | `Fwd Header Length` | `0.994879` | Perfectly redundant mathematical duplicate |
| `Total Fwd Packets` | `Bwd Header Length` | `0.998170` | Perfectly redundant mathematical duplicate |
| `Total Backward Packets` | `Bwd Header Length` | `0.999039` | Perfectly redundant mathematical duplicate |
| `Total Length of Bwd Packets` | `Bwd Header Length` | `0.997568` | Perfectly redundant mathematical duplicate |
| `Fwd Header Length` | `Bwd Header Length` | `0.997787` | Perfectly redundant mathematical duplicate |

---

## 6. Feature Relationship with Target (Feature Importance Rankings)

Feature utility for distinguishing **BENIGN (0)** from **ATTACK (1)** traffic was evaluated using **Random Forest Gini Importance**, **ANOVA F-score**, and **Mutual Information**:

### Top 20 Most Predictive Features

| Rank | Feature Name | RF Importance | Mutual Information | ANOVA F-Score | Preliminary Assessment |
| ---: | :--- | ---: | ---: | ---: | :--- |
| 1 | `Packet Length Std` | 0.08047 | 0.32639 | 98165.8 | Tier 2 - Possible candidate |
| 2 | `Avg Bwd Segment Size` | 0.07522 | 0.28909 | 110326.5 | Tier 3 - Remove/Avoid |
| 3 | `Packet Length Variance` | 0.07322 | 0.32560 | 80945.8 | Tier 2 - Possible candidate |
| 4 | `Max Packet Length` | 0.06726 | 0.27119 | 89537.3 | Tier 2 - Possible candidate |
| 5 | `Bwd Packet Length Max` | 0.05489 | 0.26401 | 111707.7 | Tier 2 - Possible candidate |
| 6 | `Bwd Packet Length Std` | 0.05478 | 0.22603 | 121081.4 | Tier 2 - Possible candidate |
| 7 | `Average Packet Size` | 0.04329 | 0.33525 | 74841.4 | Tier 3 - Remove/Avoid |
| 8 | `Total Length of Bwd Packets` | 0.02848 | 0.29671 | 0.9 | Tier 3 - Remove/Avoid |
| 9 | `Bwd Packet Length Mean` | 0.02688 | 0.28863 | 110326.5 | Tier 3 - Remove/Avoid |
| 10 | `Destination Port` | 0.02569 | 0.25885 | 5282.0 | Tier 2 - Possible candidate (Use with caution) |
| 11 | `Packet Length Mean` | 0.02510 | 0.30921 | 74623.2 | Tier 3 - Remove/Avoid |
| 12 | `Subflow Fwd Packets` | 0.02122 | 0.11327 | 2.5 | Tier 3 - Remove/Avoid |
| 13 | `Total Fwd Packets` | 0.02045 | 0.11423 | 2.5 | Tier 3 - Remove/Avoid |
| 14 | `Subflow Fwd Bytes` | 0.02018 | 0.26261 | 200.6 | Tier 3 - Remove/Avoid |
| 15 | `Subflow Bwd Bytes` | 0.01808 | 0.29759 | 0.9 | Tier 3 - Remove/Avoid |
| 16 | `Fwd IAT Max` | 0.01733 | 0.21980 | 58374.3 | Tier 3 - Remove/Avoid |
| 17 | `Total Length of Fwd Packets` | 0.01557 | 0.26252 | 200.6 | Tier 3 - Remove/Avoid |
| 18 | `PSH Flag Count` | 0.01519 | 0.01162 | 4079.5 | Tier 1 - Strong candidate |
| 19 | `Fwd IAT Std` | 0.01438 | 0.16921 | 69887.6 | Tier 2 - Possible candidate |
| 20 | `Fwd Header Length` | 0.01420 | 0.18969 | 2.4 | Tier 3 - Remove/Avoid |

---

## 7. Potential Data Leakage Analysis

Features were audited for potential laboratory environment data leakage:

1. **`Destination Port` (HIGH LEAKAGE RISK):**
   * Encodes specific service ports used during attack simulation (e.g. Port `80` for Web/DoS Hulk, Port `21` for FTP-Patator, Port `22` for SSH-Patator, Port `8080`).
   * *Risk:* Models trained on `Destination Port` may memorize synthetic testbed port numbers rather than general network anomaly dynamics.
2. **`Flow Duration` (MEDIUM LEAKAGE RISK):**
   * Attack execution scripts ran for fixed time windows (e.g. exactly 60 seconds or 300 seconds), creating duration clustering artifacts.
3. **`Fwd Header Length` & `Bwd Header Length` (MEDIUM LEAKAGE RISK):**
   * Contains tool-specific integer overflow artifacts created by CICFlowMeter under heavy load.

---

## 8. GAN Synthetic Data Generation Suitability

Features were evaluated for continuous flow generation using Generative Adversarial Networks (GANs):

* **High GAN Suitability:** Bounded, continuous payload length statistics (`Max Packet Length`, `Packet Length Std`, `Average Packet Size`, `Fwd Packet Length Mean`).
* **Needs Preprocessing:** Highly skewed features with heavy zero-inflation (`Flow Duration`, `Flow IAT Max`, `Active Mean`, `Idle Mean`). Requires `log1p` transformation and Robust Scaling.
* **Low GAN Suitability (Avoid for Direct GAN Generation):** Constant features (`Bwd PSH Flags`, `Fwd Avg Bytes/Bulk`), integer counters (`Total Fwd Packets`), and discrete port numbers (`Destination Port`).

---

## Recommendations for Final 32-Feature Selection

> [!TIP]
> This section is formatted specifically for downstream model evaluation and feature selection.

### Tier 1 — Strong Candidates (19 Features)
*Recommended for inclusion in final feature set due to high predictive power, non-redundancy, and sound network semantics:*

* **`Fwd Packet Length Min`** (RF: `0.0035`, MI: `0.0952`): Minimum payload length of forward packets.
* **`Bwd Packet Length Min`** (RF: `0.0067`, MI: `0.0911`): Minimum payload length of backward packets.
* **`Flow Bytes/s`** (RF: `0.0090`, MI: `0.1892`): Number of flow payload bytes transferred per second.
* **`Flow IAT Min`** (RF: `0.0029`, MI: `0.0615`): Minimum Inter-Arrival Time between packets in the flow.
* **`Fwd IAT Min`** (RF: `0.0058`, MI: `0.0609`): Minimum Inter-Arrival Time of forward packets.
* **`Bwd IAT Total`** (RF: `0.0014`, MI: `0.1513`): Total Inter-Arrival Time of packets sent in the backward direction.
* **`Bwd IAT Std`** (RF: `0.0014`, MI: `0.1116`): Standard deviation of Inter-Arrival Time of backward packets.
* **`Bwd IAT Max`** (RF: `0.0032`, MI: `0.1665`): Maximum Inter-Arrival Time of backward packets.
* **`Bwd Packets/s`** (RF: `0.0102`, MI: `0.1797`): Number of backward packets sent per second.
* **`Min Packet Length`** (RF: `0.0039`, MI: `0.0956`): Minimum payload length of any packet in the flow.
* **`FIN Flag Count`** (RF: `0.0019`, MI: `0.0278`): Number of packets with the FIN (Finish) flag set.
* **`PSH Flag Count`** (RF: `0.0152`, MI: `0.0116`): Number of packets with the PSH (Push) flag set.
* **`ACK Flag Count`** (RF: `0.0105`, MI: `0.0088`): Number of packets with the ACK (Acknowledgment) flag set.
* **`Down/Up Ratio`** (RF: `0.0018`, MI: `0.0238`): Ratio of backward packets to forward packets in the flow.
* **`Init_Win_bytes_forward`** (RF: `0.0105`, MI: `0.2479`): Total bytes sent in initial TCP window in forward direction (-1 if non-TCP).
* **`Init_Win_bytes_backward`** (RF: `0.0084`, MI: `0.2638`): Total bytes sent in initial TCP window in backward direction (-1 if non-TCP).
* **`min_seg_size_forward`** (RF: `0.0110`, MI: `0.0358`): Minimum segment size observed in forward direction (bytes).
* **`Active Max`** (RF: `0.0010`, MI: `0.1474`): Maximum time flow was active before going idle.
* **`Idle Std`** (RF: `0.0004`, MI: `0.0224`): Standard deviation of idle duration.


### Tier 2 — Possible Candidates (21 Features)
*Features requiring further consideration due to moderate correlation or potential environment leakage:*

* **`Destination Port`**: No high correlation. Leakage Risk: `High`. The destination port number of the network flow connection.
* **`Fwd Packet Length Max`**: Max corr 0.97 with Fwd Packet Length Std. Leakage Risk: `Low`. Maximum payload length of forward packets.
* **`Fwd Packet Length Std`**: Max corr 0.97 with Fwd Packet Length Max. Leakage Risk: `Low`. Standard deviation of payload length of forward packets.
* **`Bwd Packet Length Max`**: Max corr 0.98 with Bwd Packet Length Std. Leakage Risk: `Low`. Maximum payload length of backward packets.
* **`Bwd Packet Length Std`**: Max corr 0.98 with Bwd Packet Length Max. Leakage Risk: `Low`. Standard deviation of payload length of backward packets.
* **`Flow Packets/s`**: Max corr 0.98 with Fwd Packets/s. Leakage Risk: `Low`. Number of flow packets transferred per second.
* **`Flow IAT Mean`**: Max corr 0.90 with Fwd IAT Mean. Leakage Risk: `Low`. Mean Inter-Arrival Time between any two consecutive packets in the flow.
* **`Flow IAT Std`**: Max corr 0.94 with Flow IAT Max. Leakage Risk: `Low`. Standard deviation of Inter-Arrival Time between packets in the flow.
* **`Fwd IAT Mean`**: Max corr 0.90 with Flow IAT Mean. Leakage Risk: `Low`. Mean Inter-Arrival Time of packets sent in the forward direction.
* **`Fwd IAT Std`**: Max corr 0.92 with Fwd IAT Max. Leakage Risk: `Low`. Standard deviation of Inter-Arrival Time of forward packets.
* **`Bwd IAT Mean`**: Max corr 0.93 with Bwd IAT Min. Leakage Risk: `Low`. Mean Inter-Arrival Time of packets sent in the backward direction.
* **`Bwd IAT Min`**: Max corr 0.93 with Bwd IAT Mean. Leakage Risk: `Low`. Minimum Inter-Arrival Time of backward packets.
* **`Fwd Packets/s`**: Max corr 0.98 with Flow Packets/s. Leakage Risk: `Low`. Number of forward packets sent per second.
* **`Max Packet Length`**: Max corr 0.98 with Packet Length Std. Leakage Risk: `Low`. Maximum payload length of any packet in the flow.
* **`Packet Length Std`**: Max corr 0.98 with Max Packet Length. Leakage Risk: `Low`. Standard deviation of packet payload lengths in the flow.
* **`Packet Length Variance`**: Max corr 0.93 with Packet Length Std. Leakage Risk: `Low`. Variance of packet payload lengths in the flow.
* **`URG Flag Count`**: No high correlation. Leakage Risk: `Low`. Number of packets with the URG (Urgent) flag set.
* **`Active Mean`**: Max corr 0.94 with Active Min. Leakage Risk: `Low`. Mean time flow was active before going idle.
* **`Active Std`**: No high correlation. Leakage Risk: `Low`. Standard deviation of active duration.
* **`Active Min`**: Max corr 0.94 with Active Mean. Leakage Risk: `Low`. Minimum time flow was active before going idle.
* **`Idle Min`**: Max corr 0.99 with Idle Mean. Leakage Risk: `Low`. Minimum time flow was idle before becoming active.


### Tier 3 — Remove / Avoid (38 Features)
*Features recommended for exclusion due to zero variance (constant) or 100% correlation redundancy:*

* **`Flow Duration`**: Duplicate/Extremely Redundant with Fwd IAT Total. (Total duration of the network flow in microseconds.)
* **`Total Fwd Packets`**: Duplicate/Extremely Redundant with Subflow Fwd Packets. (Total number of packets sent in the forward direction.)
* **`Total Backward Packets`**: Duplicate/Extremely Redundant with Subflow Bwd Packets. (Total number of packets sent in the backward direction.)
* **`Total Length of Fwd Packets`**: Duplicate/Extremely Redundant with Subflow Fwd Bytes. (Total size of packet payloads sent in the forward direction (bytes).)
* **`Total Length of Bwd Packets`**: Duplicate/Extremely Redundant with Subflow Bwd Bytes. (Total size of packet payloads sent in the backward direction (bytes).)
* **`Fwd Packet Length Mean`**: Duplicate/Extremely Redundant with Avg Fwd Segment Size. (Mean payload length of forward packets.)
* **`Bwd Packet Length Mean`**: Duplicate/Extremely Redundant with Avg Bwd Segment Size. (Mean payload length of backward packets.)
* **`Flow IAT Max`**: Duplicate/Extremely Redundant with Fwd IAT Max. (Maximum Inter-Arrival Time between packets in the flow.)
* **`Fwd IAT Total`**: Duplicate/Extremely Redundant with Flow Duration. (Total Inter-Arrival Time of packets sent in the forward direction.)
* **`Fwd IAT Max`**: Duplicate/Extremely Redundant with Flow IAT Max. (Maximum Inter-Arrival Time of forward packets.)
* **`Fwd PSH Flags`**: Duplicate/Extremely Redundant with SYN Flag Count. (Number of forward packets with the PSH (Push) flag set.)
* **`Bwd PSH Flags`**: Constant (Zero Variance). (Number of backward packets with the PSH (Push) flag set.)
* **`Fwd URG Flags`**: Duplicate/Extremely Redundant with CWE Flag Count. (Number of forward packets with the URG (Urgent) flag set.)
* **`Bwd URG Flags`**: Constant (Zero Variance). (Number of backward packets with the URG (Urgent) flag set.)
* **`Fwd Header Length`**: Duplicate/Extremely Redundant with Fwd Header Length.1. (Total header bytes of packets in the forward direction.)
* **`Bwd Header Length`**: Duplicate/Extremely Redundant with Total Backward Packets. (Total header bytes of packets in the backward direction.)
* **`Packet Length Mean`**: Duplicate/Extremely Redundant with Average Packet Size. (Mean payload length of all packets in the flow.)
* **`SYN Flag Count`**: Duplicate/Extremely Redundant with Fwd PSH Flags. (Number of packets with the SYN (Synchronize) flag set.)
* **`RST Flag Count`**: Duplicate/Extremely Redundant with ECE Flag Count. (Number of packets with the RST (Reset) flag set.)
* **`CWE Flag Count`**: Duplicate/Extremely Redundant with Fwd URG Flags. (Number of packets with the CWE (Congestion Window Reduced) flag set.)
* **`ECE Flag Count`**: Duplicate/Extremely Redundant with RST Flag Count. (Number of packets with the ECN-Echo (ECE) flag set.)
* **`Average Packet Size`**: Duplicate/Extremely Redundant with Packet Length Mean. (Average payload size of packets in the flow.)
* **`Avg Fwd Segment Size`**: Duplicate/Extremely Redundant with Fwd Packet Length Mean. (Average segment size observed in forward direction.)
* **`Avg Bwd Segment Size`**: Duplicate/Extremely Redundant with Bwd Packet Length Mean. (Average segment size observed in backward direction.)
* **`Fwd Header Length.1`**: Duplicate/Extremely Redundant with Fwd Header Length. (Duplicate feature of forward header length (redundant column).)
* **`Fwd Avg Bytes/Bulk`**: Constant (Zero Variance). (Average number of bytes transferred per bulk in forward direction.)
* **`Fwd Avg Packets/Bulk`**: Constant (Zero Variance). (Average number of packets transferred per bulk in forward direction.)
* **`Fwd Avg Bulk Rate`**: Constant (Zero Variance). (Average bulk transfer rate in forward direction.)
* **`Bwd Avg Bytes/Bulk`**: Constant (Zero Variance). (Average number of bytes transferred per bulk in backward direction.)
* **`Bwd Avg Packets/Bulk`**: Constant (Zero Variance). (Average number of packets transferred per bulk in backward direction.)
* **`Bwd Avg Bulk Rate`**: Constant (Zero Variance). (Average bulk transfer rate in backward direction.)
* **`Subflow Fwd Packets`**: Duplicate/Extremely Redundant with Total Fwd Packets. (Average number of packets in a forward subflow.)
* **`Subflow Fwd Bytes`**: Duplicate/Extremely Redundant with Total Length of Fwd Packets. (Average number of bytes in a forward subflow.)
* **`Subflow Bwd Packets`**: Duplicate/Extremely Redundant with Total Backward Packets. (Average number of packets in a backward subflow.)
* **`Subflow Bwd Bytes`**: Duplicate/Extremely Redundant with Total Length of Bwd Packets. (Average number of bytes in a backward subflow.)
* **`act_data_pkt_fwd`**: Duplicate/Extremely Redundant with Total Backward Packets. (Number of forward packets with at least 1 byte of TCP payload.)
* **`Idle Mean`**: Duplicate/Extremely Redundant with Idle Max. (Mean time flow was idle before becoming active.)
* **`Idle Max`**: Duplicate/Extremely Redundant with Idle Mean. (Maximum time flow was idle before becoming active.)

---

## 9. Generated Artifacts & Reproducibility

All outputs have been automatically generated and saved:

1. **Cleaned Dataset:** `data/cleaned/cleaned_cic_ids2017.parquet` & `data/cleaned/cleaned_cic_ids2017.csv`
2. **Combined Raw Dataset:** `data/combined/combined_raw_cic_ids2017.parquet` & `data/combined/combined_raw_cic_ids2017.csv`
3. **Machine-Readable Reports:** `reports/feature_analysis.csv`, `reports/feature_analysis.json`, `reports/dataset_summary.csv`
4. **Visualizations:** `reports/figures/01_label_distribution.png` through `06_attack_vs_benign_comparison.png`
5. **Python Pipeline Script:** `analyze_cic_ids2017.py`

*Analysis completed in `812.14` seconds.*
