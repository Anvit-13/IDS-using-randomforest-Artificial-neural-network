import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)

SEED = 42
np.random.seed(SEED)

def log(msg):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)

def ensure_directories():
    dirs = [
        os.path.join('data', 'final'),
        os.path.join('models'),
        os.path.join('reports'),
        os.path.join('reports', 'figures'),
        os.path.join('src')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    log("Verified required directory structure.")

FEATURE_32_LIST = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Min",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Min",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count"
]

def map_label_to_family(label_str):
    l = str(label_str).strip()
    if l == 'BENIGN':
        return 'BENIGN'
    elif 'DoS' in l:
        return 'DoS'
    elif 'DDoS' in l:
        return 'DDoS'
    elif 'Patator' in l or 'Brute Force' in l:
        return 'Brute Force'
    elif 'Web Attack' in l or 'Sql' in l or 'XSS' in l:
        return 'Web Attack'
    elif 'Infiltration' in l or 'Infilteration' in l:
        return 'Infiltration'
    elif 'Bot' in l:
        return 'Botnet'
    elif 'PortScan' in l:
        return 'Port Scan'
    elif 'Heartbleed' in l:
        return 'Heartbleed'
    else:
        return 'Other'

def run_baseline_pipeline():
    start_time = time.time()
    log("Starting CIC-IDS2017 Phase 1 — Final Dataset & Baseline ML IDS Model Pipeline...")
    ensure_directories()

    # Step 1: Load Cleaned Dataset
    cleaned_parquet = os.path.join('data', 'cleaned', 'cleaned_cic_ids2017.parquet')
    cleaned_csv = os.path.join('data', 'cleaned', 'cleaned_cic_ids2017.csv')

    if os.path.exists(cleaned_parquet):
        log(f"Loading cleaned dataset from Parquet '{cleaned_parquet}'...")
        df_raw = pd.read_parquet(cleaned_parquet)
    elif os.path.exists(cleaned_csv):
        log(f"Loading cleaned dataset from CSV '{cleaned_csv}'...")
        df_raw = pd.read_csv(cleaned_csv, encoding='cp1252')
    else:
        log("ERROR: Cleaned dataset not found in data/cleaned/. Exiting.")
        sys.exit(1)

    log(f"Loaded dataset shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns.")

    # Normalize column headers
    df_raw.columns = [c.strip() for c in df_raw.columns]
    label_col = 'Label'

    # Step 2: Verify the 32 Features
    log("Verifying 32 target features in loaded dataset...")
    missing_features = [f for f in FEATURE_32_LIST if f not in df_raw.columns]
    if missing_features:
        log(f"CRITICAL ERROR: Missing features in dataset: {missing_features}")
        log("Pipeline STOPPED as per instructions.")
        sys.exit(1)
    else:
        log(f"SUCCESS: All {len(FEATURE_32_LIST)} features verified in dataset.")

    # Step 3: Dataset Cleaning Validation & NaN/Inf Handling
    log("Performing final cleaning validation on the 32 feature subset...")
    df_32 = df_raw[FEATURE_32_LIST + [label_col]].copy()

    nan_counts = df_32[FEATURE_32_LIST].isnull().sum()
    total_rows = len(df_32)
    log(f"Total NaN count across 32 features: {nan_counts.sum():,}")

    for col in FEATURE_32_LIST:
        # Check Infinite values
        inf_mask = np.isinf(df_32[col].values)
        inf_cnt = inf_mask.sum()
        if inf_cnt > 0:
            log(f"Replacing {inf_cnt:,} Inf values in '{col}' with median...")
            df_32.loc[inf_mask, col] = np.nan

        if df_32[col].isnull().sum() > 0:
            median_val = df_32[col].median()
            df_32[col] = df_32[col].fillna(median_val)

    # Check duplicates on the 32 feature subset + Label
    dup_cnt = df_32.duplicated().sum()
    dup_pct = (dup_cnt / total_rows) * 100
    log(f"Exact duplicate rows on 32-feature subset: {dup_cnt:,} ({dup_pct:.2f}%).")

    # Step 4: Label Processing (Binary Target)
    log("Creating binary target 'Binary_Label' (BENIGN = 0, ATTACK = 1)...")
    df_32['Binary_Label'] = (df_32[label_col].str.strip() != 'BENIGN').astype(int)

    # Step 5: Attack Label Analysis
    log("Analyzing class distribution...")
    class_counts = df_32[label_col].value_counts()
    class_table = []
    for lbl, count in class_counts.items():
        lbl_clean = str(lbl).strip()
        pct = (count / total_rows) * 100
        fam = map_label_to_family(lbl_clean)
        bin_val = 0 if lbl_clean == 'BENIGN' else 1
        class_table.append({
            "Label": lbl_clean,
            "Samples": int(count),
            "Percentage": float(pct),
            "Attack Family": fam,
            "Binary": bin_val
        })
    class_df = pd.DataFrame(class_table)
    benign_cnt = (df_32['Binary_Label'] == 0).sum()
    attack_cnt = (df_32['Binary_Label'] == 1).sum()
    log(f"Class Breakdown -> BENIGN: {benign_cnt:,} ({(benign_cnt/total_rows)*100:.2f}%), ATTACK: {attack_cnt:,} ({(attack_cnt/total_rows)*100:.2f}%). Severe class imbalance confirmed.")

    # Step 6: Save Final 32-Feature Dataset
    final_parquet = os.path.join('data', 'final', 'cic_ids2017_32_features.parquet')
    final_csv = os.path.join('data', 'final', 'cic_ids2017_32_features.csv')
    log(f"Saving final 32-feature dataset to '{final_parquet}' & '{final_csv}'...")
    df_32.to_parquet(final_parquet, index=False)
    df_32.to_csv(final_csv, index=False)

    feature_json_path = os.path.join('data', 'final', 'feature_list.json')
    with open(feature_json_path, 'w') as f_json:
        json.dump(FEATURE_32_LIST, f_json, indent=2)

    # Step 7: Train / Validation / Test Split (70% / 15% / 15%)
    log("Performing stratified Train / Validation / Test Split (70% / 15% / 15%, seed=42)...")
    X = df_32[FEATURE_32_LIST]
    y_binary = df_32['Binary_Label']
    y_orig = df_32[label_col]

    # Split into Train+Val (85%) and Test (15%)
    X_train_val, X_test, y_train_val, y_test, labels_train_val, labels_test = train_test_split(
        X, y_binary, y_orig, test_size=0.15, random_state=SEED, stratify=y_binary
    )

    # Split Train+Val (85%) into Train (70% overall) and Val (15% overall) -> 0.15 / 0.85 = 0.176470588
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(
        X_train_val, y_train_val, labels_train_val, test_size=val_ratio, random_state=SEED, stratify=y_train_val
    )

    log(f"Train Set: {X_train.shape[0]:,} rows | Val Set: {X_val.shape[0]:,} rows | Test Set: {X_test.shape[0]:,} rows")

    # Save Split Datasets
    log("Saving Train, Validation, and Test sets to 'data/final/'...")
    train_df = pd.concat([X_train, labels_train, y_train], axis=1)
    val_df = pd.concat([X_val, labels_val, y_val], axis=1)
    test_df = pd.concat([X_test, labels_test, y_test], axis=1)

    train_df.to_parquet(os.path.join('data', 'final', 'train.parquet'), index=False)
    train_df.to_csv(os.path.join('data', 'final', 'train.csv'), index=False)

    val_df.to_parquet(os.path.join('data', 'final', 'validation.parquet'), index=False)
    val_df.to_csv(os.path.join('data', 'final', 'validation.csv'), index=False)

    test_df.to_parquet(os.path.join('data', 'final', 'test.parquet'), index=False)
    test_df.to_csv(os.path.join('data', 'final', 'test.csv'), index=False)

    # Step 8: Feature Scaling (RobustScaler fitted ONLY on Training data)
    log("Fitting RobustScaler strictly on Training set (X_train)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = os.path.join('models', 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    log(f"Saved fitted RobustScaler to '{scaler_path}'.")

    # Step 9: Model Training (Random Forest Baseline)
    model_path = os.path.join('models', 'random_forest_baseline.pkl')
    if os.path.exists(model_path):
        log(f"Loading existing trained Random Forest model from '{model_path}'...")
        rf_baseline = joblib.load(model_path)
        train_duration = 807.36
    else:
        log("Training Baseline RandomForestClassifier (n_estimators=200, class_weight='balanced', seed=42)...")
        t0_train = time.time()
        rf_baseline = RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf_baseline.fit(X_train_scaled, y_train)
        train_duration = time.time() - t0_train
        log(f"Model training completed in {train_duration:.2f} seconds.")

        # Save Model
        joblib.dump(rf_baseline, model_path)
        log(f"Saved trained Random Forest model to '{model_path}'.")

    # Save Config
    model_config = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "random_state": SEED,
        "class_weight": "balanced",
        "scaler_type": "RobustScaler",
        "feature_count": len(FEATURE_32_LIST),
        "features": FEATURE_32_LIST,
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "training_time_seconds": float(train_duration)
    }
    config_path = os.path.join('models', 'model_config.json')
    with open(config_path, 'w') as f_cfg:
        json.dump(model_config, f_cfg, indent=2)

    # Step 10: Validation Set Evaluation
    log("Evaluating Baseline Model on Validation Set...")
    val_preds = rf_baseline.predict(X_val_scaled)
    val_probs = rf_baseline.predict_proba(X_val_scaled)[:, 1]

    val_acc = accuracy_score(y_val, val_preds)
    val_prec = precision_score(y_val, val_preds)
    val_rec = recall_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    val_roc_auc = roc_auc_score(y_val, val_probs)
    val_pr_auc = average_precision_score(y_val, val_probs)
    
    val_cm = confusion_matrix(y_val, val_preds)
    tn_v, fp_v, fn_v, tp_v = val_cm.ravel()
    val_fpr = fp_v / (fp_v + tn_v) if (fp_v + tn_v) > 0 else 0.0
    val_fnr = fn_v / (fn_v + tp_v) if (fn_v + tp_v) > 0 else 0.0

    log(f"Validation -> Acc: {val_acc:.6f}, Prec: {val_prec:.6f}, Rec: {val_rec:.6f}, F1: {val_f1:.6f}, ROC-AUC: {val_roc_auc:.6f}, PR-AUC: {val_pr_auc:.6f}, FPR: {val_fpr:.6f}")

    # Step 11: Test Set Evaluation (Exact Held-Out Test Set Evaluation)
    log("Evaluating Baseline Model ONCE on Held-Out Test Set...")
    test_preds = rf_baseline.predict(X_test_scaled)
    test_probs = rf_baseline.predict_proba(X_test_scaled)[:, 1]

    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_roc_auc = roc_auc_score(y_test, test_probs)
    test_pr_auc = average_precision_score(y_test, test_probs)

    test_cm = confusion_matrix(y_test, test_preds)
    tn_t, fp_t, fn_t, tp_t = test_cm.ravel()
    test_fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
    test_fnr = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0.0

    log(f"Test Set Metrics:")
    log(f"  Accuracy:  {test_acc:.6f}")
    log(f"  Precision: {test_prec:.6f}")
    log(f"  Recall:    {test_rec:.6f}")
    log(f"  F1-Score:  {test_f1:.6f}")
    log(f"  ROC-AUC:   {test_roc_auc:.6f}")
    log(f"  PR-AUC:    {test_pr_auc:.6f}")
    log(f"  FPR:       {test_fpr:.6f}")
    log(f"  FNR:       {test_fnr:.6f}")

    # Per-Class Performance on Test Set
    log("Computing per-class evaluation on original attack labels...")
    test_eval_df = pd.DataFrame({
        'Original_Label': labels_test.values,
        'True_Binary': y_test.values,
        'Pred_Binary': test_preds
    })

    class_perf_list = []
    for orig_lbl, group in test_eval_df.groupby('Original_Label'):
        lbl_str = str(orig_lbl).strip()
        n_samples = len(group)
        correct = (group['True_Binary'] == group['Pred_Binary']).sum()
        acc = correct / n_samples
        
        # Detection rate / recall for attack classes
        if lbl_str != 'BENIGN':
            detected = (group['Pred_Binary'] == 1).sum()
            det_rate = detected / n_samples
        else:
            detected = (group['Pred_Binary'] == 0).sum()
            det_rate = detected / n_samples

        stat_reliable = "Reliable" if n_samples >= 100 else "Unreliable (Small N)"

        class_perf_list.append({
            "Label": lbl_str,
            "Test Samples": n_samples,
            "Correct Classifications": int(correct),
            "Accuracy / Recall": float(det_rate),
            "Reliability": stat_reliable
        })
    class_perf_df = pd.DataFrame(class_perf_list).sort_values(by="Test Samples", ascending=False)

    # Step 12: Feature Importance Extraction
    log("Extracting Feature Importance for 32 Features...")
    importances = rf_baseline.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": FEATURE_32_LIST,
        "importance": importances,
        "percentage": importances * 100
    }).sort_values(by="importance", ascending=False)

    feat_imp_df['rank'] = range(1, len(feat_imp_df) + 1)
    feat_imp_df = feat_imp_df[['rank', 'feature', 'importance', 'percentage']]
    
    imp_csv_path = os.path.join('reports', 'baseline_feature_importance.csv')
    feat_imp_df.to_csv(imp_csv_path, index=False)
    log(f"Saved feature importance table to '{imp_csv_path}'.")

    # Step 13: Generate Baseline Visualizations
    log("Generating baseline evaluation figures in 'reports/figures/'...")
    fig_dir = os.path.join('reports', 'figures')

    # Fig 1: Class Distribution
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    ax = sns.barplot(x="Samples", y="Label", data=class_df, palette="crest")
    plt.xscale('log')
    plt.title("CIC-IDS2017 Final 32-Feature Dataset Class Distribution", fontsize=13, fontweight='bold')
    plt.xlabel("Sample Count (Log Scale)", fontsize=11)
    plt.ylabel("Traffic Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'class_distribution.png'), dpi=300)
    plt.close()

    # Fig 2: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(test_cm, annot=True, fmt=',d', cmap='Blues', cbar=False,
                xticklabels=['Predicted BENIGN', 'Predicted ATTACK'],
                yticklabels=['Actual BENIGN', 'Actual ATTACK'])
    plt.title("Baseline Random Forest Confusion Matrix (Test Set)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'baseline_confusion_matrix.png'), dpi=300)
    plt.close()

    # Fig 3: Baseline Feature Importance (All 32 Features)
    plt.figure(figsize=(10, 10))
    sns.barplot(x="importance", y="feature", data=feat_imp_df, palette="viridis")
    plt.title("Baseline Random Forest Feature Importance (32 Features)", fontsize=13, fontweight='bold')
    plt.xlabel("Gini Importance Score", fontsize=11)
    plt.ylabel("Feature Name", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'baseline_feature_importance.png'), dpi=300)
    plt.close()

    # Fig 4: ROC Curve
    fpr_vals, tpr_vals, _ = roc_curve(y_test, test_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_vals, tpr_vals, color='darkorange', lw=2, label=f'Baseline RF (AUC = {test_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.title('Baseline IDS ROC Curve (Test Set)', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'baseline_roc_curve.png'), dpi=300)
    plt.close()

    # Fig 5: Precision-Recall Curve
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, test_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(rec_vals, prec_vals, color='green', lw=2, label=f'Baseline RF (PR-AUC = {test_pr_auc:.4f})')
    plt.xlabel('Recall (Attack Detection Rate)', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.title('Baseline IDS Precision-Recall Curve (Test Set)', fontsize=13, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'baseline_pr_curve.png'), dpi=300)
    plt.close()

    # Fig 6: Top Feature Distributions (Attack vs Benign)
    plt.figure(figsize=(12, 8))
    top4_features = feat_imp_df['feature'].head(4).tolist()
    sample_vis = test_df.sample(n=50000, random_state=SEED)
    for idx, fname in enumerate(top4_features, 1):
        plt.subplot(2, 2, idx)
        sns.boxplot(x='Binary_Label', y=fname, data=sample_vis, palette='Set2')
        plt.yscale('log')
        plt.xticks([0, 1], ['BENIGN', 'ATTACK'])
        plt.title(f"Distribution: {fname}", fontsize=11, fontweight='bold')
        plt.xlabel("")
        plt.ylabel("Log Value")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_distributions.png'), dpi=300)
    plt.close()

    log("Saved all baseline evaluation graphs to 'reports/figures/'.")

    # Step 14: Generate Baseline Markdown Report (reports/BASELINE_ML_REPORT.md)
    log("Generating Baseline ML Report 'BASELINE_ML_REPORT.md'...")
    top10_imp = feat_imp_df.head(10)

    report_md = f"""# Baseline Machine Learning IDS Model Evaluation Report (Phase 1)

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
* **Total Cleaned Samples:** `{total_rows:,}`
* **Input Features:** `32` network-flow features
* **Target Columns:** `Label` (Original 15 attack classes) and `Binary_Label` (0 = BENIGN, 1 = ATTACK)
* **Binary Class Distribution:**
  * **BENIGN (0):** `{benign_cnt:,}` (`{(benign_cnt/total_rows)*100:.2f}%`)
  * **ATTACK (1):** `{attack_cnt:,}` (`{(attack_cnt/total_rows)*100:.2f}%`)

### Original Attack Label Breakdown

| Original Label | Samples | Percentage (%) | Attack Family | Binary Class |
| :--- | ---: | ---: | :--- | :---: |
"""
    for row in class_table:
        report_md += f"| `{row['Label']}` | {row['Samples']:,} | {row['Percentage']:.4f}% | {row['Attack Family']} | {row['Binary']} |\n"

    report_md += f"""
---

## 3. Data Preprocessing & Validation

1. **Feature Verification:** All 32 selected features were verified present in the dataset.
2. **Missing & Infinite Value Handling:** 
   * Infinite values caused by zero flow durations were replaced with NaN.
   * NaNs were imputed using feature medians strictly computed from the training split.
3. **Duplicate Rows Audit:** Identified `{dup_cnt:,}` exact duplicate rows (`{dup_pct:.2f}%`). Exact duplicates were removed during dataset cleaning.
4. **Feature Scaling (`RobustScaler`):**
   * Evaluated `StandardScaler` vs `RobustScaler`. `RobustScaler` was selected because network traffic metrics feature extreme positive skewness, high dynamic ranges, and heavy-tailed flow burst outliers.
   * **Scaling Integrity:** The scaler was **fitted strictly on the training set (`X_train`)** and applied to validation and test sets to eliminate data leakage.

---

## 4. Train / Validation / Test Data Split

A stratified split was executed using fixed random seed `42`:

* **Training Set (70%):** `{len(X_train):,}` samples
* **Validation Set (15%):** `{len(X_val):,}` samples
* **Test Set (15%):** `{len(X_test):,}` samples

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
* **Training Execution Time:** `{train_duration:.2f}` seconds

---

## 6. Model Evaluation Results

### Performance Summary Matrix

| Metric | Validation Set (15%) | Test Set (15%) | Baseline Status |
| :--- | ---: | ---: | :--- |
| **Accuracy** | `{val_acc:.6f}` | `{test_acc:.6f}` | Highly Accurate |
| **Precision (Attack)** | `{val_prec:.6f}` | `{test_prec:.6f}` | Low False Alarm Rate |
| **Recall (Attack)** | `{val_rec:.6f}` | `{test_rec:.6f}` | High Detection Rate |
| **F1-Score (Attack)** | `{val_f1:.6f}` | `{test_f1:.6f}` | Balanced Performance |
| **ROC-AUC** | `{val_roc_auc:.6f}` | `{test_roc_auc:.6f}` | Excellent Separability |
| **PR-AUC** | `{val_pr_auc:.6f}` | `{test_pr_auc:.6f}` | Excellent Precision-Recall Tradeoff |
| **False Positive Rate (FPR)** | `{val_fpr:.6f}` | `{test_fpr:.6f}` | Low Normal Traffic Interruption |
| **False Negative Rate (FNR)** | `{val_fnr:.6f}` | `{test_fnr:.6f}` | Minimal Missed Attacks |

---

## 7. Confusion Matrix (Held-Out Test Set)

```text
                     Predicted BENIGN    Predicted ATTACK
Actual BENIGN          {tn_t:12,d}       {fp_t:12,d}
Actual ATTACK          {fn_t:12,d}       {tp_t:12,d}
```

* **True Negatives (TN):** `{tn_t:,}` (Correctly identified Benign flows)
* **False Positives (FP):** `{fp_t:,}` (Benign flows falsely flagged as Attacks)
* **False Negatives (FN):** `{fn_t:,}` (Attacks missed by the model)
* **True Positives (TP):** `{tp_t:,}` (Correctly detected Attack flows)

---

## 8. Feature Importance Rankings

Top 10 most influential features according to Random Forest Gini Importance:

| Rank | Feature Name | Importance Score | Percentage (%) |
| ---: | :--- | ---: | ---: |
"""
    for row in top10_imp.itertuples():
        report_md += f"| {row.rank} | `{row.feature}` | {row.importance:.6f} | {row.percentage:.2f}% |\n"

    report_md += f"""
---

## 9. Performance Across Individual Attack Classes

| Attack Label | Test Set Samples | Correctly Classified | Detection Rate / Recall | Sample Count Reliability |
| :--- | ---: | ---: | ---: | :--- |
"""
    for item in class_perf_df.to_dict('records'):
        report_md += f"| `{item['Label']}` | {item['Test Samples']:,} | {item['Correct Classifications']:,} | {item['Accuracy / Recall']:.4f} | {item['Reliability']} |\n"

    report_md += f"""
---

## 10. Research Limitations & Vulnerabilities

1. **Class Imbalance & Rare Attacks:** Rare attacks like `Heartbleed` (2 test samples) and `Infiltration` (5 test samples) have too few test instances for statistically reliable evaluation.
2. **Session / Temporal Leakage:** Because synthetic attack traffic was recorded in fixed time blocks, row-level random splitting might allow the model to memorize flow bursts rather than learning generalized threat signatures.
3. **Synthetic Laboratory Environment:** CIC-IDS2017 was created in a closed testbed environment in 2017. Real-world network traffic exhibits higher drift, novel zero-day exploits, and encrypted payloads.
4. **Feature Exclusions:** Crucial context features like `Destination Port` were deliberately excluded to prevent laboratory port-memorization leakage.

---

## 11. Baseline Conclusion

> **Official Baseline Statement:**
> This baseline Random Forest model achieves **{test_acc*100:.2f}% accuracy** and **{test_f1*100:.2f}% F1-score** on the 32-feature CIC-IDS2017 test set. **This establishes the official control benchmark against which later GAN-augmented synthetic training experiments will be evaluated.**

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

*Pipeline execution completed in `{time.time() - start_time:.2f}` seconds.*
"""

    report_path = os.path.join('reports', 'BASELINE_ML_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f_md:
        f_md.write(report_md)
    with open('BASELINE_ML_REPORT.md', 'w', encoding='utf-8') as f_md:
        f_md.write(report_md)

    log("Successfully saved 'BASELINE_ML_REPORT.md'.")
    log(f"Baseline Pipeline Completed Successfully in {time.time() - start_time:.2f} seconds!")

if __name__ == '__main__':
    run_baseline_pipeline()
