# SynTabGAN: Public Cloud AI/ML Research Platform

> **Design and Development of a Working Prototype for Synthetic Tabular Data Generator Using a Generative Adversarial Network for Tabular Data**  
> *Final-Year B.Tech Engineering Project Demonstration & Streamlit Community Cloud Public Release*

[![Streamlit Cloud Ready](https://img.shields.io/badge/Streamlit%20Cloud-Deployment%20Ready-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20CPU-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SDV CTGAN](https://img.shields.io/badge/SDV-CTGAN%202.x-1E3E62.svg?style=flat)](https://github.com/sdv-dev/SDV)
[![Tests Passing](https://img.shields.io/badge/pytest-27%20passed-10B981.svg?style=flat)](tests/)

---

## 📌 1. Project Overview & Commercial AI Platform Architecture
**SynTabGAN** is a research-grade web platform for synthesizing realistic, privacy-conscious tabular data. It bridges the gap between strict data privacy compliance (GDPR, HIPAA, DPDP) and the need for data-sharing in analytics and machine learning.

Built on **SDV (Synthetic Data Vault)** and **PyTorch**, the platform trains a **Conditional Tabular GAN (CTGAN)** featuring:
- **Mode-Specific Normalization (VGM)**: Variational Gaussian Mixtures with $K=10$ modes for continuous multimodal columns.
- **Conditional Generator**: Balances rare discrete categories via conditioned sampling masks.
- **PacGAN Discriminator**: Packs 10 consecutive records to enforce Wasserstein gradient dynamics and prevent mode collapse.
- **SDMetrics & DCR Auditing**: Computes empirical fidelity (Column Shapes, Pair Trends) and disclosure risk (Distance to Closest Record).

---

## 📁 2. Production Repository Structure

```
Synthetic_Tabular_Data_Generator/
├── app.py                      # Main entry point for Streamlit Community Cloud
├── requirements.txt            # Cloud-optimized dependency specifications
├── README.md                   # Comprehensive deployment and architecture documentation
├── .gitignore                  # Strict rules excluding virtualenvs, caches, and user data
├── .python-version             # Specifies Python 3.11.0 runtime for cloud builders
├── .streamlit/
│   └── config.toml             # Headless server, CORS, max 10MB upload, and navy theme
├── assets/
│   ├── logo.svg                # Vector AI synthesis logo
│   └── styles.css              # Custom Dark Navy & Light Blue theme CSS
├── modules/
│   ├── __init__.py
│   ├── data_loader.py          # CSV loading, validation, profiling, and type detection
│   ├── preprocessor.py         # Missing value imputation, deduplication, SDV metadata
│   ├── gan_model.py            # SDV CTGANSynthesizer & PyTorch WGAN-GP training wrapper
│   ├── evaluator.py            # SDMetrics QualityReport, KS-2samp, TVD, correlation MAE
│   └── privacy_analyzer.py     # Exact match detection, Distance to Closest Record (DCR)
├── utils/
│   ├── __init__.py
│   ├── config.py               # Constants, theme colors, and cloud resource limits
│   ├── visualizer.py           # Interactive Plotly chart builders & 2D PCA manifold
│   └── report_generator.py     # Self-contained HTML/Markdown evaluation report compiler
├── sample_data/
│   ├── employee_attrition.csv       # HR Analytics (150 rows)
│   ├── loan_risk_assessment.csv     # FinTech / Credit Risk (150 rows)
│   └── patient_health_records.csv   # Healthcare Diagnostics (150 rows)
├── models/
│   └── .gitkeep                # Directory for local model checkpoints
├── pages/
│   └── __init__.py             # Multi-page modular namespace
├── tests/
│   ├── test_data_loader.py     # Ingestion & profiling unit tests
│   ├── test_preprocessor.py    # Preprocessing & SDV metadata unit tests
│   ├── test_gan_model.py       # CTGAN model fitting & generation tests
│   ├── test_evaluator.py       # SDMetrics & KS-test unit tests
│   ├── test_privacy_analyzer.py# Empirical privacy & DCR unit tests
│   └── test_e2e_scenarios.py   # Comprehensive 19-scenario end-to-end test suite
├── run.bat                     # Windows 1-click execution script
└── run.sh                      # Linux/macOS launcher script
```

---

## ☁️ 3. Streamlit Community Cloud Deployment Guide

Streamlit Community Cloud provides free hosting for public GitHub repositories. Follow these exact steps:

### Step 1: Create a GitHub Repository
1. Log in to your GitHub account at [github.com](https://github.com).
2. Click **New Repository** (or visit [github.com/new](https://github.com/new)).
3. Name the repository: `Synthetic-Tabular-Data-Generator`.
4. Set visibility to **Public** (required for free Streamlit Cloud deployment).
5. Leave "Add a README file" and ".gitignore" unchecked (our local repository already contains production-ready versions).
6. Click **Create repository**.

### Step 2: Push Local Code to GitHub
Open your terminal (PowerShell or Git Bash) in `c:\Users\vigne\Downloads\Synthetic_Tabular_Data_Generator` and run:

```bash
# Initialize git repository
git init -b main

# Add all files (respecting .gitignore)
git add .

# Create initial release commit
git commit -m "feat: production release of SynTabGAN platform for Streamlit Cloud"

# Link to your remote GitHub repository (replace with your GitHub username)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/Synthetic-Tabular-Data-Generator.git

# Push code to main branch
git push -u origin main
```

### Step 3: Deploy on Streamlit Community Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click the **"New app"** button.
3. Configure the deployment settings:
   - **Repository:** `<YOUR_GITHUB_USERNAME>/Synthetic-Tabular-Data-Generator`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** Custom name (e.g., `syntabgan.streamlit.app`)
4. Click **"Deploy!"**
5. Streamlit Cloud will read `requirements.txt` and `.python-version` (Python 3.11), install dependencies, and launch the platform.
6. Once deployed, the public URL will be accessible to faculty and evaluators globally.

---

## ⚡ 4. Cloud Resource Optimizations & Defensive Safeguards

Streamlit Community Cloud containers operate under a **1 GB RAM limit**. To prevent container evictions or Out-Of-Memory (OOM) failures:

1. **Upload File Guard:** Enforces a strict `10 MB` max upload limit via `.streamlit/config.toml`.
2. **Dynamic Row Downsampling:** Datasets exceeding `5,000 rows` display a cloud advisory alert with a 1-click downsample button to `3,000 rows`.
3. **PacGAN Batch Divisibility:** Batch sizes and `pac` values are dynamically adjusted to eliminate tensor shape mismatch errors on small or odd-sized datasets.
4. **Garbage Collection:** Explicit `gc.collect()` calls are triggered after dataset ingestion, model fitting, and sample generation to release freed memory buffers.
5. **CPU-Optimized PyTorch:** Model parameters use lightweight MLP structures with `enable_gpu=False` fallback, training 35 epochs in under 15 seconds.

---

## 🧪 5. Automated Verification & Quality Assurance Report

The project includes 27 comprehensive automated tests executed with `pytest`:

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### Verification Matrix (All 27 Tests Passed)

| Scenario / Test Case | Scope | Status |
| :--- | :--- | :---: |
| `test_scenario_01_valid_csv` | Ingestion of standard tabular CSV | **PASSED** |
| `test_scenario_02_invalid_file` | Rejection of non-CSV corrupt binary files | **PASSED** |
| `test_scenario_03_missing_values` | Detection and percentage profiling of NaN cells | **PASSED** |
| `test_scenario_04_duplicate_records` | Accurate identification of duplicate rows | **PASSED** |
| `test_scenario_05_numerical_only` | Auto-detection and handling of purely numeric schemas | **PASSED** |
| `test_scenario_06_categorical_only` | Auto-detection and handling of purely discrete schemas | **PASSED** |
| `test_scenario_07_mixed_data` | Handling of heterogeneous numeric and discrete schemas | **PASSED** |
| `test_scenario_08_preprocessing` | Median/Mode imputation and duplicate stripping | **PASSED** |
| `test_scenario_09_to_12_generation` | Sampling 100, 500, and 1,000 records from trained CTGAN | **PASSED** |
| `test_scenario_13_quality_evaluation`| SDMetrics composite score, KS-test, and TVD distance | **PASSED** |
| `test_scenario_14_privacy_analysis` | Distance to Closest Record (DCR) and exact match rates | **PASSED** |
| `test_scenario_15_visualizations` | Interactive Plotly distributions, heatmaps, and PCA | **PASSED** |
| `test_scenario_16_csv_export` | Encoding and validation of synthetic CSV downloads | **PASSED** |
| `test_scenario_17_html_report_export`| Generation of standalone executive HTML audit reports | **PASSED** |
| `test_edge_empty_dataset` | Graceful failure on 0-byte or header-only files | **PASSED** |
| `test_edge_single_column_dataset` | Single-feature training and distribution modeling | **PASSED** |
| `test_edge_unsupported_data_types` | Special characters, emojis, and irregular text strings | **PASSED** |
| `test_edge_small_dataset` | Robust execution on ultra-small datasets (5 rows) | **PASSED** |
| `test_edge_large_dataset_within_mem`| Scaling validation on 1,500 rows $\times$ 10 columns | **PASSED** |
| `test_data_loader.py` (3 tests) | Ingestion and profiling unit coverage | **PASSED** |
| `test_evaluator.py` (1 test) | SDMetrics statistical fidelity unit coverage | **PASSED** |
| `test_gan_model.py` (1 test) | SDV CTGANSynthesizer integration unit coverage | **PASSED** |
| `test_preprocessor.py` (2 tests) | Imputation and SDV metadata builder coverage | **PASSED** |
| `test_privacy_analyzer.py` (1 test)| Nearest-neighbor distance calculation coverage | **PASSED** |

---

## 💻 6. Local Development & Demo Commands

### Windows One-Click
Double click `run.bat` or run:
```cmd
run.bat
```

### Manual CLI Run
```powershell
# Activate local environment
.\.venv\Scripts\activate.ps1

# Run Streamlit dashboard
streamlit run app.py
```

---

## 🛡️ 7. Scientific & Ethical Privacy Notice
Empirical metrics like Distance to Closest Record (DCR) evaluate separation from training samples. They **do not** confer mathematical **Differential Privacy ($\epsilon, \delta$-DP)**. Adversaries with auxiliary background information can still theoretically attempt membership inference. Synthetic data should be classified as **pseudonymous** rather than unconditionally anonymous.

---

## 🎓 Academic Project Credits
- **Title:** Design and Development of a Working Prototype for Synthetic Tabular Data Generator Using a Generative Adversarial Network for Tabular Data
- **Degree:** Bachelor of Technology (B.Tech) - Final Year Engineering Project
- **Core Libraries:** PyTorch, SDV 2.x, CTGAN, SDMetrics, Streamlit, Plotly, Scikit-learn
