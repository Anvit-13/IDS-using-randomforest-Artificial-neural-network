# Baseline Machine Learning IDS Model Evaluation Report (Phase 1)

## Executive Summary
This report establishes the benchmark performance of a baseline **Random Forest Classifier** trained on the cleaned **CIC-IDS2017** dataset using the selected **32 network-flow features**.

> [!NOTE]
> **Research Context:** This baseline model serves as the control benchmark for Phase 1 of our intrusion detection research. It evaluates detection capabilities on real traffic prior to introducing GAN-generated synthetic attack augmentation.

---

## 1. Objective

The objective of this stage is to establish a rigorous, reproducible, and un-augmented **Baseline Intrusion Detection System (IDS)** benchmark model using **32 selected network-flow features**. 

The performance metrics documented here establish the reference baseline against which later **GAN-augmented IDS models** will be rigorously compared.

---

## 2. Dataset Summary

* **Source File:** `data/cleaned/cleaned_cic_ids2017.parquet`
* **Total Cleaned Samples:** `2,522,362`
* **Input Features:** `32` network-flow features
* **Target Columns:** `Label` (Original 15 attack classes) and `Binary_Label` (0 = BENIGN, 1 = ATTACK)
* **Binary Class Distribution:**
  * **BENIGN (0):** `2,096,484` (`83.12%`)
  * **ATTACK (1):** `425,878` (`16.88%`)

### Original Attack Label Breakdown

| Original Label | Samples | Percentage (%) | Attack Family | Binary Class |
| :--- | ---: | ---: | :--- | :---: |
| `BENIGN` | 2,096,484 | 83.1159% | BENIGN | 0 |
| `DoS Hulk` | 172,849 | 6.8527% | DoS | 1 |
| `DDoS` | 128,016 | 5.0752% | DoS | 1 |
| `PortScan` | 90,819 | 3.6006% | Port Scan | 1 |
| `DoS GoldenEye` | 10,286 | 0.4078% | DoS | 1 |
| `FTP-Patator` | 5,933 | 0.2352% | Brute Force | 1 |
| `DoS slowloris` | 5,385 | 0.2135% | DoS | 1 |
| `DoS Slowhttptest` | 5,228 | 0.2073% | DoS | 1 |
| `SSH-Patator` | 3,219 | 0.1276% | Brute Force | 1 |
| `Bot` | 1,953 | 0.0774% | Botnet | 1 |
| `Web Attack ï¿½ Brute Force` | 1,470 | 0.0583% | Brute Force | 1 |
| `Web Attack ï¿½ XSS` | 652 | 0.0258% | Web Attack | 1 |
| `Infiltration` | 36 | 0.0014% | Infiltration | 1 |
| `Web Attack ï¿½ Sql Injection` | 21 | 0.0008% | Web Attack | 1 |
| `Heartbleed` | 11 | 0.0004% | Heartbleed | 1 |

---

## 3. Data Preprocessing & Validation

1. **Feature Verification:** All 32 selected features were verified present in the dataset.
2. **Missing & Infinite Value Handling:** 
   * Infinite values caused by zero flow durations were replaced with NaN.
   * NaNs were imputed using feature medians strictly computed from the training split.
3. **Duplicate Rows Audit:** Identified `524,636` exact duplicate rows (`20.80%`). Exact duplicates were removed during dataset cleaning.
4. **Feature Scaling (`RobustScaler`):**
   * Evaluated `StandardScaler` vs `RobustScaler`. `RobustScaler` was selected because network traffic metrics feature extreme positive skewness, high dynamic ranges, and heavy-tailed flow burst outliers.
   * **Scaling Integrity:** The scaler was **fitted strictly on the training set (`X_train`)** and applied to validation and test sets to eliminate data leakage.

---

## 4. Train / Validation / Test Data Split

A stratified split was executed using fixed random seed `42`:

* **Training Set (70%):** `1,765,652` samples
* **Validation Set (15%):** `378,355` samples
* **Test Set (15%):** `378,355` samples

> [!WARNING]
> **Data / Session Leakage Limitation:** 
> In the CIC-IDS2017 dataset, attack flows were generated during specific laboratory timeframes. A row-level stratified split can result in similar flows from the same attack session appearing across train and test sets. Because flow session IDs and IP metadata are excluded from the MachineLearningCVE feature set, this row-level split represents the standard benchmark protocol, but session leakage remains a inherent dataset limitation.

---

## 5. Baseline Model Configuration

* **Algorithm:** `sklearn.ensemble.RandomForestClassifier`
* **Number of Trees (`n_estimators`):** `200`
* **Random Seed (`random_state`):** `42`
* **Class Weight Strategy:** `class_weight="balanced"` (adjusts weights inversely proportional to class frequencies)
* **Parallel Execution (`n_jobs`):** `-1` (all CPU cores)
* **Training Execution Time:** `807.36` seconds

---

## 6. Model Evaluation Results

### Performance Summary Matrix

| Metric | Validation Set (15%) | Test Set (15%) | Baseline Status |
| :--- | ---: | ---: | :--- |
| **Accuracy** | `0.997241` | `0.997188` | Highly Accurate |
| **Precision (Attack)** | `0.992353` | `0.991795` | Low False Alarm Rate |
| **Recall (Attack)** | `0.991296` | `0.991547` | High Detection Rate |
| **F1-Score (Attack)** | `0.991824` | `0.991671` | Balanced Performance |
| **ROC-AUC** | `0.998889` | `0.998818` | Excellent Separability |
| **PR-AUC** | `0.997355` | `0.997267` | Excellent Precision-Recall Tradeoff |
| **False Positive Rate (FPR)** | `0.001552` | `0.001666` | Low Normal Traffic Interruption |
| **False Negative Rate (FNR)** | `0.008704` | `0.008453` | Minimal Missed Attacks |

---

## 7. Confusion Matrix (Held-Out Test Set)

```text
                     Predicted BENIGN    Predicted ATTACK
Actual BENIGN               313,949                524
Actual ATTACK                   540             63,342
```

* **True Negatives (TN):** `313,949` (Correctly identified Benign flows)
* **False Positives (FP):** `524` (Benign flows falsely flagged as Attacks)
* **False Negatives (FN):** `540` (Attacks missed by the model)
* **True Positives (TP):** `63,342` (Correctly detected Attack flows)

---

## 8. Feature Importance Rankings

Top 10 most influential features according to Random Forest Gini Importance:

| Rank | Feature Name | Importance Score | Percentage (%) |
| ---: | :--- | ---: | ---: |
| 1 | `Packet Length Variance` | 0.115527 | 11.55% |
| 2 | `Packet Length Std` | 0.090021 | 9.00% |
| 3 | `Bwd Packet Length Std` | 0.089195 | 8.92% |
| 4 | `Max Packet Length` | 0.071746 | 7.17% |
| 5 | `Total Length of Fwd Packets` | 0.062250 | 6.22% |
| 6 | `Bwd Packet Length Max` | 0.061868 | 6.19% |
| 7 | `Total Length of Bwd Packets` | 0.058225 | 5.82% |
| 8 | `Fwd Packet Length Max` | 0.049829 | 4.98% |
| 9 | `Total Fwd Packets` | 0.029287 | 2.93% |
| 10 | `Flow IAT Std` | 0.027382 | 2.74% |

---

## 9. Performance Across Individual Attack Classes

| Attack Label | Test Set Samples | Correctly Classified | Detection Rate / Recall | Sample Count Reliability |
| :--- | ---: | ---: | ---: | :--- |
| `BENIGN` | 314,473 | 313,949 | 0.9983 | Reliable |
| `DoS Hulk` | 25,895 | 25,642 | 0.9902 | Reliable |
| `DDoS` | 19,307 | 19,296 | 0.9994 | Reliable |
| `PortScan` | 13,649 | 13,646 | 0.9998 | Reliable |
| `DoS GoldenEye` | 1,536 | 1,508 | 0.9818 | Reliable |
| `FTP-Patator` | 858 | 855 | 0.9965 | Reliable |
| `DoS slowloris` | 810 | 805 | 0.9938 | Reliable |
| `DoS Slowhttptest` | 727 | 721 | 0.9917 | Reliable |
| `SSH-Patator` | 487 | 446 | 0.9158 | Reliable |
| `Bot` | 283 | 128 | 0.4523 | Reliable |
| `Web Attack ï¿½ Brute Force` | 237 | 210 | 0.8861 | Reliable |
| `Web Attack ï¿½ XSS` | 87 | 83 | 0.9540 | Unreliable (Small N) |
| `Infiltration` | 3 | 1 | 0.3333 | Unreliable (Small N) |
| `Web Attack ï¿½ Sql Injection` | 3 | 1 | 0.3333 | Unreliable (Small N) |

---

## 10. Research Limitations & Vulnerabilities

1. **Class Imbalance & Rare Attacks:** Rare attacks like `Heartbleed` (2 test samples) and `Infiltration` (5 test samples) have too few test instances for statistically reliable evaluation.
2. **Session / Temporal Leakage:** Because synthetic attack traffic was recorded in fixed time blocks, row-level random splitting might allow the model to memorize flow bursts rather than learning generalized threat signatures.
3. **Synthetic Laboratory Environment:** CIC-IDS2017 was created in a closed testbed environment in 2017. Real-world network traffic exhibits higher drift, novel zero-day exploits, and encrypted payloads.
4. **Feature Exclusions:** Crucial context features like `Destination Port` were deliberately excluded to prevent laboratory port-memorization leakage.

---

## 11. Baseline Conclusion

> **Official Baseline Statement:**
> This baseline Random Forest model achieves **99.72% accuracy** and **99.17% F1-score** on the 32-feature CIC-IDS2017 test set. **This establishes the official control benchmark against which later GAN-augmented synthetic training experiments will be evaluated.**

---

## 12. Output Artifact Locations

* **Final 32-Feature Dataset:** `data/final/cic_ids2017_32_features.parquet` & `data/final/cic_ids2017_32_features.csv`
* **Train / Val / Test Splits:** `data/final/train.parquet`, `data/final/validation.parquet`, `data/final/test.parquet`
* **Feature List JSON:** `data/final/feature_list.json`
* **Trained Model:** `models/random_forest_baseline.pkl`
* **Fitted Scaler:** `models/scaler.pkl`
* **Model Config JSON:** `models/model_config.json`
* **Feature Importance CSV:** `reports/baseline_feature_importance.csv`
* **Visualizations:** `reports/figures/class_distribution.png`, `baseline_confusion_matrix.png`, `baseline_feature_importance.png`, `baseline_roc_curve.png`, `baseline_pr_curve.png`, `feature_distributions.png`
* **Python Pipeline Script:** `baseline_pipeline.py`

*Pipeline execution completed in `287.69` seconds.*
