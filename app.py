"""
SynTabGAN: Public Cloud & Research Platform Edition
Streamlit Entry Point: app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import gc
from pathlib import Path

# Dual module import (supports modules/ or src/)
try:
    from modules.data_loader import load_csv, profile_dataset, detect_column_types
    from modules.preprocessor import Preprocessor, build_sdv_metadata
    from modules.gan_model import CTGANModel
    from modules.evaluator import DataQualityEvaluator
    from modules.privacy_analyzer import PrivacyRiskAnalyzer
    from utils.config import (
        THEME, DEFAULT_GAN_CONFIG, SAMPLE_DATASETS, LOGO_SVG,
        MAX_CLOUD_ROWS, SAFE_SAMPLE_ROWS, MAX_CLOUD_EPOCHS, DEFAULT_CLOUD_EPOCHS
    )
    from utils.visualizer import (
        plot_numerical_distribution,
        plot_categorical_distribution,
        plot_correlation_heatmaps,
        plot_pca_projection,
        plot_dcr_histogram
    )
    from utils.report_generator import generate_html_report
except ImportError:
    from src.data_loader import load_csv, profile_dataset, detect_column_types
    from src.preprocessor import Preprocessor, build_sdv_metadata
    from src.gan_model import CTGANModel
    from src.evaluator import DataQualityEvaluator
    from src.privacy_analyzer import PrivacyRiskAnalyzer
    from src.config import THEME, DEFAULT_GAN_CONFIG, SAMPLE_DATASETS, LOGO_SVG
    from src.visualizer import (
        plot_numerical_distribution,
        plot_categorical_distribution,
        plot_correlation_heatmaps,
        plot_pca_projection,
        plot_dcr_histogram
    )
    from src.report_generator import generate_html_report
    MAX_CLOUD_ROWS = 5000
    SAFE_SAMPLE_ROWS = 3000
    MAX_CLOUD_EPOCHS = 100
    DEFAULT_CLOUD_EPOCHS = 35

# ---------------------------------------------------------
# Page Configuration & Styles
# ---------------------------------------------------------
st.set_page_config(
    page_title="SynTabGAN | Generative Tabular AI Research Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = Path(__file__).resolve().parent / "assets" / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "None"
if "preprocessed_df" not in st.session_state:
    st.session_state.preprocessed_df = None
if "preprocessor_audit" not in st.session_state:
    st.session_state.preprocessor_audit = None
if "discrete_columns" not in st.session_state:
    st.session_state.discrete_columns = []
if "ctgan_model" not in st.session_state:
    st.session_state.ctgan_model = None
if "training_status" not in st.session_state:
    st.session_state.training_status = None
if "synthetic_df" not in st.session_state:
    st.session_state.synthetic_df = None
if "quality_results" not in st.session_state:
    st.session_state.quality_results = None
if "privacy_results" not in st.session_state:
    st.session_state.privacy_results = None

# ---------------------------------------------------------
# Sidebar Navigation & Branding
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand">
        {LOGO_SVG}
        <div>
            <div class="sidebar-brand-name">SynTabGAN</div>
            <div class="sidebar-brand-sub">SDV CTGAN Research Lab</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "CTGAN Architecture & Theory",
            "Dataset Upload",
            "Data Preprocessing",
            "GAN Training",
            "Synthetic Data Generation",
            "Quality Evaluation",
            "Privacy Analysis",
            "Analytics Dashboard",
            "Download Reports",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚡ Curated Sample Datasets")
    selected_demo = st.selectbox(
        "Load Demonstration Data",
        list(SAMPLE_DATASETS.keys()),
        index=0
    )
    if st.button("🚀 Load Selected Dataset", use_container_width=True):
        demo_path = SAMPLE_DATASETS[selected_demo]
        try:
            loaded_df = pd.read_csv(demo_path)
            st.session_state.raw_df = loaded_df
            st.session_state.dataset_name = selected_demo.split(" (")[0]
            detected = detect_column_types(loaded_df)
            st.session_state.discrete_columns = detected["categorical"]
            # Clear downstream states & collect garbage
            st.session_state.preprocessed_df = None
            st.session_state.ctgan_model = None
            st.session_state.synthetic_df = None
            st.session_state.quality_results = None
            st.session_state.privacy_results = None
            gc.collect()
            st.success(f"Loaded '{st.session_state.dataset_name}' successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading demo dataset: {str(e)}")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 11px; color: #94A3B8; text-align: center;">
            <b>Final-Year B.Tech Project</b><br>
            Public Cloud Production Build<br>
            Optimized for Streamlit Cloud
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Top Header & Dynamic Metric Cards
# ---------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <div>
        <h1 class="brand-title">Synthetic Tabular Data Generator</h1>
        <p class="brand-subtitle">Conditional Tabular Generative Adversarial Network (SDV CTGAN) &bull; SDMetrics Quality &bull; DCR Privacy</p>
    </div>
    <div class="badge-pill">
        <span>⚡ SDV CTGAN Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

raw_rows = len(st.session_state.raw_df) if st.session_state.raw_df is not None else 0
raw_cols = st.session_state.raw_df.shape[1] if st.session_state.raw_df is not None else 0
raw_missing = int(st.session_state.raw_df.isna().sum().sum()) if st.session_state.raw_df is not None else 0
synth_rows = len(st.session_state.synthetic_df) if st.session_state.synthetic_df is not None else 0

if st.session_state.quality_results is not None:
    quality_display = f"{st.session_state.quality_results.get('overall_score', 0)}%"
else:
    quality_display = "Pending"

if st.session_state.ctgan_model is not None and st.session_state.ctgan_model.is_trained:
    training_status = "Trained ✅"
    training_sub = f"{st.session_state.ctgan_model.training_time_sec}s runtime"
else:
    training_status = "Untrained ⏳"
    training_sub = "Awaiting fit"

st.markdown(f"""
<div class="metric-container">
    <div class="research-card">
        <div class="card-accent-strip"></div>
        <div class="card-label">Original Rows</div>
        <div class="card-value">{raw_rows:,}</div>
        <div class="card-subtext">{st.session_state.dataset_name}</div>
    </div>
    <div class="research-card">
        <div class="card-accent-strip"></div>
        <div class="card-label">Total Columns</div>
        <div class="card-value">{raw_cols}</div>
        <div class="card-subtext">Features profile</div>
    </div>
    <div class="research-card">
        <div class="card-accent-strip"></div>
        <div class="card-label">Missing Values</div>
        <div class="card-value">{raw_missing}</div>
        <div class="card-subtext">Cells to impute</div>
    </div>
    <div class="research-card">
        <div class="card-accent-strip"></div>
        <div class="card-label">Synthetic Rows</div>
        <div class="card-value">{synth_rows:,}</div>
        <div class="card-subtext">Generated samples</div>
    </div>
    <div class="research-card">
        <div class="card-accent-strip"></div>
        <div class="card-label">Quality Score</div>
        <div class="card-value">{quality_display}</div>
        <div class="card-subtext">SDMetrics composite</div>
    </div>
    <div class="research-card">
        <div class="card-accent-strip"></div>
        <div class="card-label">GAN Status</div>
        <div class="card-value" style="font-size: 18px; margin-top: 4px;">{training_status}</div>
        <div class="card-subtext">{training_sub}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cloud Memory Guard Alert
# ---------------------------------------------------------
if raw_rows > MAX_CLOUD_ROWS:
    st.warning(
        f"⚠️ **Cloud Resource Notice**: The loaded dataset contains {raw_rows:,} rows. "
        f"To prevent memory exhaustion on cloud containers (1GB RAM limit), we recommend sampling {SAFE_SAMPLE_ROWS:,} rows for smooth GAN training."
    )
    if st.button(f"✂️ Downsample Dataset to {SAFE_SAMPLE_ROWS:,} Rows", key="btn_downsample"):
        st.session_state.raw_df = st.session_state.raw_df.sample(SAFE_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
        gc.collect()
        st.success(f"Dataset downsampled to {SAFE_SAMPLE_ROWS:,} rows.")
        st.rerun()

# =========================================================
# PAGE 1: HOME
# =========================================================
if page == "Home":
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">🏛️ Project Overview & Generative Modeling Blueprint</div>
        <p style="color: #475569; font-size: 15px; line-height: 1.7;">
            Welcome to the <b>SynTabGAN Research Platform</b>. This system is a final-year B.Tech engineering prototype 
            designed to solve data privacy bottlenecks and training data scarcity using 
            <b>Conditional Tabular Generative Adversarial Networks (CTGAN)</b> from the Synthetic Data Vault (SDV).
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">🔬 Key CTGAN Algorithmic Innovations</div>
            <ul style="color: #334155; font-size: 14px; line-height: 1.8;">
                <li><b>Mode-Specific Normalization:</b> Continuous numerical columns with non-Gaussian and multimodal distributions are encoded using a Variational Gaussian Mixture Model (VGM), overcoming the collapse seen in standard GANs.</li>
                <li><b>Conditional Generator:</b> Enables conditional sampling on discrete categorical attributes, ensuring balanced representation across minority categories and mitigating class imbalance.</li>
                <li><b>PacGAN Discriminator:</b> Feeds stacked samples (packs) to the discriminator to penalize mode collapse and stabilize Wasserstein GAN gradient dynamics.</li>
                <li><b>Rigorous Dual Auditing:</b> Evaluates both <i>Statistical Fidelity</i> (SDMetrics, KS-test, Total Variation Distance) and <i>Empirical Privacy Risk</i> (Distance to Closest Record - DCR).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">🗺️ End-to-End Workflow Pipeline</div>
            <div style="background: #F1F5F9; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 13px; color: #0F172A;">
                [1. Ingestion] ➔ [2. Preprocessing & SDV Metadata] ➔ [3. SDV CTGAN Training]<br>
                ➔ [4. Sampling] ➔ [5. SDMetrics Quality Audit] ➔ [6. Privacy DCR Analysis] ➔ [7. HTML Report]
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">🚀 Getting Started</div>
            <p style="font-size: 13px; color: #475569;">
                Choose how you want to demonstrate the application:
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.raw_df is None:
            st.info("💡 No dataset loaded yet. Choose an action below:")
            if st.button("⚡ Load Demo HR Analytics Data", use_container_width=True):
                demo_df = pd.read_csv(SAMPLE_DATASETS["Employee Attrition (HR Analytics)"])
                st.session_state.raw_df = demo_df
                st.session_state.dataset_name = "Employee Attrition"
                detected = detect_column_types(demo_df)
                st.session_state.discrete_columns = detected["categorical"]
                st.success("Loaded Employee Attrition dataset!")
                st.rerun()
        else:
            st.success(f"Active dataset: **{st.session_state.dataset_name}** ({len(st.session_state.raw_df)} rows, {st.session_state.raw_df.shape[1]} cols)")
            st.markdown("Navigate through the sidebar to explore CTGAN architecture, train the model, and generate synthetic records.")

# =========================================================
# PAGE 2: CTGAN ARCHITECTURE & THEORY
# =========================================================
elif page == "CTGAN Architecture & Theory":
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">🧬 Technical Explanation: Conditional Tabular GAN (CTGAN) Deep-Dive</div>
        <p style="color: #475569; font-size: 14px; line-height: 1.7;">
            This technical reference page details the mathematical mechanics, internal neural networks, 
            adversarial training dynamics, and empirical limitations of CTGAN as formulated by Lei Xu et al. (NeurIPS 2019).
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("""
        <div class="theory-card">
            <h4>1. The Generator Network ($G$)</h4>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                <b>Role:</b> Maps a stochastic Gaussian latent vector $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I}_{128})$ 
                concatenated with an active conditional category vector $\mathbf{cond}$ into synthetic continuous and discrete feature vectors.<br><br>
                <b>Architecture:</b> Multi-layer perceptron (MLP) with Residual skips.
                Continuous features are output as mode-representation scalars ($\tanh$) along with mode assignment probability vectors ($\text{softmax}$).
                Discrete features are output as categorical probability distributions ($\text{softmax}$).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="theory-card">
            <h4>2. The Discriminator Network ($D$)</h4>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                <b>Role:</b> Acts as an adversarial critic measuring the Wasserstein distance between real tabular data distribution $\mathbb{P}_{\text{data}}$ 
                and generator distribution $\mathbb{P}_{\text{syn}}$.<br><br>
                <b>PacGAN Innovation:</b> Standard tabular discriminators suffer severe mode collapse. CTGAN employs <b>PacGAN</b> ($pac=10$), 
                stacking $10$ consecutive rows into a single input tensor. This forces the discriminator to evaluate joint distributions over minibatches rather than individual rows.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="theory-card">
            <h4>3. Mode-Specific Normalization (VGM)</h4>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                Tabular continuous columns rarely follow unimodal Gaussian distributions. CTGAN fits a <b>Variational Gaussian Mixture Model (VGM)</b> with $K=10$ modes:
                $$\mathbb{P}(c) = \sum_{k=1}^K w_k \mathcal{N}(c; \mu_k, \sigma_k^2)$$
                Each scalar value $c$ is decomposed into a mode assignment one-hot vector $\mathbf{\beta} \in \{0, 1\}^K$ 
                and a normalized scalar offset within the mode $v = \frac{c - \mu_k}{4\sigma_k} \in [-1, 1]$.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown("""
        <div class="theory-card">
            <h4>4. Adversarial Training Process (WGAN-GP)</h4>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                The objective is formulated using the Kantorovich-Rubinstein duality with Gradient Penalty:
                $$\mathcal{L}_D = -\mathbb{E}[D(\mathbf{X}_{\text{real}})] + \mathbb{E}[D(\mathbf{X}_{\text{fake}})] + \lambda_{\text{GP}} \mathbb{E}\left[(\|\nabla_{\hat{\mathbf{X}}} D(\hat{\mathbf{X}})\|_2 - 1)^2\right]$$
                $$\mathcal{L}_G = -\mathbb{E}[D(G(\mathbf{z}, \mathbf{cond}))] + \text{CrossEntropy}(G_{\text{cat}}, \mathbf{cond})$$
                The cross-entropy term forces $G$ to faithfully produce the exact categorical attribute specified by the condition vector.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="theory-card">
            <h4>5. CTGAN Training & Generation Flow</h4>
            <p style="font-size: 13px; color: #334155; line-height: 1.6;">
                <b>Step 1 - Data Encoding:</b> Fit VGM on numeric columns and one-hot encode discrete columns.<br>
                <b>Step 2 - Conditional Minibatches:</b> Sample condition vectors according to log-frequency of discrete columns to balance rare classes.<br>
                <b>Step 3 - Adversarial Updates:</b> Train $D$ for $1$ step, then update $G$ with WGAN-GP backpropagation.<br>
                <b>Step 4 - Sampling:</b> Sample $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$, invert normalized modes back to real values, and apply post-processing.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="theory-card" style="border-left: 4px solid #F59E0B; background: #FFFBEB;">
            <h4 style="color: #92400E;">⚠️ Scientific & Privacy Limitations Notice</h4>
            <p style="font-size: 13px; color: #92400E; line-height: 1.6;">
                <b>Crucial Research Finding:</b> High statistical fidelity (e.g. SDMetrics $>90\%$) does <b>not</b> guarantee privacy. 
                Heuristic GANs optimize distribution matching, not anonymity. Empirical metrics like Distance to Closest Record (DCR) 
                evaluate separation from training data, but cannot prevent linkage attacks under strong background knowledge. 
                Formal privacy requires mathematical <b>Differential Privacy ($\epsilon$-DP)</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# PAGE 3: DATASET UPLOAD
# =========================================================
elif page == "Dataset Upload":
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">📁 Ingest Real-World Tabular Dataset</div>
        <p style="color: #475569; font-size: 14px;">
            Upload your custom CSV dataset or inspect the currently loaded demonstration dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Tabular Dataset (CSV format, max 10MB)",
        type=["csv"],
        help="Upload standard CSV tabular files. Max 10MB file size."
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Parsing and validating CSV..."):
                loaded_df = load_csv(uploaded_file)
                st.session_state.raw_df = loaded_df
                st.session_state.dataset_name = uploaded_file.name
                detected = detect_column_types(loaded_df)
                st.session_state.discrete_columns = detected["categorical"]
                # Reset downstream
                st.session_state.preprocessed_df = None
                st.session_state.ctgan_model = None
                st.session_state.synthetic_df = None
                st.session_state.quality_results = None
                st.session_state.privacy_results = None
                gc.collect()
            st.success(f"Successfully loaded '{uploaded_file.name}' with {len(loaded_df)} rows and {loaded_df.shape[1]} columns!")
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")

    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        profile = profile_dataset(df)

        st.markdown("### 📊 Dataset Statistical Profile")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Rows Count", f"{profile['rows']:,}")
        col_m2.metric("Columns Count", f"{profile['columns']}")
        col_m3.metric("Missing Values", f"{profile['missing_cells']} ({profile['missing_pct']}%)")
        col_m4.metric("Duplicate Rows", f"{profile['duplicate_rows']} ({profile['duplicate_pct']}%)")

        tab_preview, tab_cols = st.tabs(["📋 Data Preview (First 15 Rows)", "🔍 Column Metadata & Detection"])

        with tab_preview:
            st.dataframe(df.head(15), use_container_width=True)

        with tab_cols:
            st.dataframe(profile["column_profile_df"], use_container_width=True)
            st.info(
                f"**Auto-Detected Categorical Features ({len(profile['column_classification']['categorical'])}):** "
                f"{', '.join(profile['column_classification']['categorical']) if profile['column_classification']['categorical'] else 'None'}"
            )
            if profile["column_classification"]["id_like"]:
                st.warning(
                    f"**Candidate Unique Identifier Columns ({len(profile['column_classification']['id_like'])}):** "
                    f"{', '.join(profile['column_classification']['id_like'])} (Recommended to exclude during preprocessing)."
                )
    else:
        st.markdown("""
        <div class="empty-state">
            <h3>No Dataset Uploaded</h3>
            <p>Please upload a CSV file above or choose a pre-packaged sample dataset from the left sidebar.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# PAGE 4: DATA PREPROCESSING
# =========================================================
elif page == "Data Preprocessing":
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">⚙️ Data Preprocessing & SDV Metadata Schema Configuration</div>
        <p style="color: #475569; font-size: 14px;">
            Clean missing values, deduplicate records, filter identifier columns, and finalize discrete column designations for CTGAN.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.raw_df is None:
        st.markdown("""
        <div class="empty-state">
            <h3>No Dataset to Preprocess</h3>
            <p>Upload a dataset in the 'Dataset Upload' page or load a sample from the sidebar first.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        raw_df = st.session_state.raw_df
        detected = detect_column_types(raw_df)

        c1, c2 = st.columns([1, 1])
        with c1:
            impute_strat = st.selectbox(
                "Missing Values Imputation Strategy",
                ["impute_median_mode", "impute_mean_mode", "drop_rows", "none"],
                index=0,
                help="Median/Mode is robust to tabular outliers. Drop rows deletes records with missing values."
            )
            drop_dupes = st.checkbox("Drop Duplicate Records", value=True)
            drop_ids = st.checkbox("Drop Identifier (ID/UUID) Columns", value=True)

        with c2:
            st.markdown("<b>Select Discrete / Categorical Columns for CTGAN:</b>", unsafe_allow_html=True)
            chosen_discrete = st.multiselect(
                "Categorical Columns (Targeted by Conditional Generator)",
                options=list(raw_df.columns),
                default=[c for c in detected["categorical"] if c in raw_df.columns]
            )

        if st.button("⚡ Execute Preprocessing Pipeline", use_container_width=True):
            with st.spinner("Applying cleaning transformations..."):
                try:
                    prep = Preprocessor(
                        impute_strategy=impute_strat,
                        drop_duplicates=drop_dupes,
                        drop_id_columns=drop_ids
                    )
                    clean_df, audit = prep.fit_transform(raw_df, detected)
                    st.session_state.preprocessed_df = clean_df
                    st.session_state.preprocessor_audit = audit
                    st.session_state.discrete_columns = [c for c in chosen_discrete if c in clean_df.columns]
                    # Reset downstream trained models & garbage collect
                    st.session_state.ctgan_model = None
                    st.session_state.synthetic_df = None
                    st.session_state.quality_results = None
                    st.session_state.privacy_results = None
                    gc.collect()
                    st.success("Preprocessing completed successfully!")
                except Exception as e:
                    st.error(f"Preprocessing error: {str(e)}")

        if st.session_state.preprocessed_df is not None:
            clean_df = st.session_state.preprocessed_df
            audit = st.session_state.preprocessor_audit

            st.markdown("---")
            st.markdown("### 📋 Preprocessing Audit & Verification")
            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            col_a1.metric("Cleaned Rows", f"{len(clean_df):,}", f"{len(clean_df) - len(raw_df)} rows")
            col_a2.metric("Cleaned Columns", f"{clean_df.shape[1]}", f"{clean_df.shape[1] - raw_df.shape[1]} cols")
            col_a3.metric("Remaining NaNs", f"{int(clean_df.isna().sum().sum())}")
            col_a4.metric("Active Discrete Features", f"{len(st.session_state.discrete_columns)}")

            st.dataframe(clean_df.head(10), use_container_width=True)

# =========================================================
# PAGE 5: GAN TRAINING
# =========================================================
elif page == "GAN Training":
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">🧠 Train Conditional Tabular GAN (SDV CTGANSynthesizer)</div>
        <p style="color: #475569; font-size: 14px;">
            Configure generative adversarial network hyperparameters and execute neural model training using SDV's CTGAN implementation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    target_df = st.session_state.preprocessed_df if st.session_state.preprocessed_df is not None else st.session_state.raw_df

    if target_df is None:
        st.markdown("""
        <div class="empty-state">
            <h3>No Training Data Available</h3>
            <p>Please load a dataset before configuring GAN training.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### 🛠️ Model Hyperparameters")
        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            epochs = st.slider(
                "Training Epochs",
                min_value=10,
                max_value=MAX_CLOUD_EPOCHS,
                value=DEFAULT_CLOUD_EPOCHS,
                step=5,
                help="More epochs allow learning complex multi-modal distributions. Capped at 100 for cloud stability."
            )
            batch_size = st.select_slider("Batch Size", options=[10, 20, 50, 100, 200, 500], value=100)
        with col_h2:
            pac = st.selectbox("PacGAN Packing Size (pac)", [1, 2, 5, 10], index=3, help="Groups samples fed to discriminator to prevent mode collapse.")
            embedding_dim = st.selectbox("Generator Latent Dimension", [64, 128, 256], index=1)
        with col_h3:
            gen_lr = st.select_slider("Generator Learning Rate", options=[1e-4, 2e-4, 5e-4, 1e-3], value=2e-4, format_func=lambda x: f"{x:.0e}")
            disc_lr = st.select_slider("Discriminator Learning Rate", options=[1e-4, 2e-4, 5e-4, 1e-3], value=2e-4, format_func=lambda x: f"{x:.0e}")

        # Training Execution
        st.markdown("---")
        train_btn = st.button("🔥 Start SDV CTGAN Training", use_container_width=True)

        if train_btn:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            status_text.markdown("<b>Initializing SDV SingleTable Metadata & PyTorch CTGAN networks...</b>", unsafe_allow_html=True)
            progress_bar.progress(0.15)

            discrete_cols = st.session_state.discrete_columns
            if not discrete_cols:
                discrete_cols = detect_column_types(target_df)["categorical"]

            try:
                model = CTGANModel(
                    epochs=epochs,
                    batch_size=batch_size,
                    pac=pac,
                    embedding_dim=embedding_dim,
                    generator_lr=gen_lr,
                    discriminator_lr=disc_lr
                )

                status_text.markdown("<b>Training in progress: Fitting Variational Gaussian Mixture & PacGAN Discriminator...</b>", unsafe_allow_html=True)
                progress_bar.progress(0.40)

                with st.spinner("Executing SDV CTGAN Adversarial Optimization..."):
                    train_result = model.fit(target_df, discrete_columns=discrete_cols)

                progress_bar.progress(1.0)
                status_text.markdown("<b>Training completed successfully!</b>", unsafe_allow_html=True)

                st.session_state.ctgan_model = model
                st.session_state.training_status = train_result
                # Clear older synthetic data & garbage collect
                st.session_state.synthetic_df = None
                st.session_state.quality_results = None
                st.session_state.privacy_results = None
                gc.collect()
                st.success(f"SDV CTGAN trained successfully in {train_result['training_time']} on {train_result['device'].upper()}!")
            except MemoryError:
                st.error("❌ Out of Memory Error: Dataset or batch size exceeds cloud container RAM. Please downsample the dataset or reduce batch size.")
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")

        if st.session_state.ctgan_model is not None and st.session_state.ctgan_model.is_trained:
            res = st.session_state.training_status
            st.markdown("### 🏆 Active Model Specification & Neural Layers")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Engine", res.get("engine", "SDV CTGAN"))
            col_s2.metric("Epochs Completed", res["epochs"])
            col_s3.metric("Effective Batch Size", res["batch_size"])
            col_s4.metric("Training Execution Time", res["training_time"])

            if "network_summary" in res:
                net = res["network_summary"]
                st.markdown("#### 🧠 Underlying Neural Network Architecture Details")
                st.json(net)

# =========================================================
# PAGE 6: SYNTHETIC DATA GENERATION
# =========================================================
elif page == "Synthetic Data Generation":
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">🎲 Generate Synthetic Tabular Records</div>
        <p style="color: #475569; font-size: 14px;">
            Sample synthetic records from the trained SDV CTGAN generator, preview results, and export as CSV.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.ctgan_model is None or not st.session_state.ctgan_model.is_trained:
        st.markdown("""
        <div class="empty-state">
            <h3>No Trained CTGAN Model Found</h3>
            <p>Please train the model on the 'GAN Training' page before generating synthetic records.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        orig_count = len(st.session_state.preprocessed_df if st.session_state.preprocessed_df is not None else st.session_state.raw_df)

        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            num_samples = st.number_input(
                "Number of Synthetic Records to Generate",
                min_value=10,
                max_value=5000,
                value=min(orig_count, 1000),
                step=50,
                help="Generates synthetic rows conditioned on the learned data manifold. Capped at 5,000 for cloud stability."
            )
        with col_g2:
            preserve_dtypes = st.checkbox("Preserve Original Data Types & Bounds", value=True)

        if st.button("🚀 Generate Synthetic Dataset", use_container_width=True):
            try:
                with st.spinner(f"Sampling {num_samples} records from SDV CTGAN generator..."):
                    synthetic_df = st.session_state.ctgan_model.generate(
                        num_rows=num_samples,
                        preserve_types=preserve_dtypes
                    )
                    st.session_state.synthetic_df = synthetic_df
                    # Reset evaluation caches & garbage collect
                    st.session_state.quality_results = None
                    st.session_state.privacy_results = None
                    gc.collect()
                st.success(f"Generated {len(synthetic_df)} synthetic records successfully!")
            except Exception as e:
                st.error(f"Generation error: {str(e)}")

        if st.session_state.synthetic_df is not None:
            synth_df = st.session_state.synthetic_df
            st.markdown("---")
            st.markdown("### 📋 Synthetic Dataset Preview")
            st.dataframe(synth_df.head(20), use_container_width=True)

            csv_data = synth_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Synthetic Data as CSV",
                data=csv_data,
                file_name=f"synthetic_{st.session_state.dataset_name.lower().replace(' ', '_')}_{len(synth_df)}rows.csv",
                mime="text/csv",
                use_container_width=True
            )

# =========================================================
# PAGE 7: QUALITY EVALUATION
# =========================================================
elif page == "Quality Evaluation":
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">📊 Rigorous Data Quality & Fidelity Evaluation</div>
        <p style="color: #475569; font-size: 14px;">
            Audits statistical fidelity using SDMetrics Single Table Quality Reports, Kolmogorov-Smirnov tests, and Total Variation Distance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    real_data = st.session_state.preprocessed_df if st.session_state.preprocessed_df is not None else st.session_state.raw_df
    synth_data = st.session_state.synthetic_df

    if real_data is None or synth_data is None:
        st.markdown("""
        <div class="empty-state">
            <h3>Prerequisites Incomplete</h3>
            <p>You need both real data and generated synthetic data to execute quality evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("🔍 Run Full Quality Audit", use_container_width=True):
            with st.spinner("Computing SDMetrics quality score and statistical tests..."):
                try:
                    evaluator = DataQualityEvaluator(
                        real_df=real_data,
                        synthetic_df=synth_data,
                        discrete_columns=st.session_state.discrete_columns
                    )
                    q_res = evaluator.generate_full_evaluation()
                    st.session_state.quality_results = q_res
                    gc.collect()
                    st.success("Quality audit complete!")
                except Exception as e:
                    st.error(f"Evaluation error: {str(e)}")

        if st.session_state.quality_results is not None:
            q_res = st.session_state.quality_results

            st.markdown("---")
            col_q1, col_q2, col_q3 = st.columns(3)
            col_q1.metric("Overall Quality Score", f"{q_res['overall_score']}%")
            col_q2.metric("Fidelity Grade", q_res["grade"])
            col_q3.metric("Correlation Score", f"{q_res['correlation_metrics'].get('preservation_score', 'N/A')}%")

            st.info(f"**Academic Verdict:** {q_res['verdict']}")

            tab_sdm, tab_num, tab_cat = st.tabs([
                "📋 SDMetrics Quality Dimensions",
                "📈 Numerical Features (KS-Test)",
                "📊 Categorical Features (TVD)"
            ])

            with tab_sdm:
                st.markdown("#### Quality Properties")
                st.dataframe(q_res["sdmetrics"]["properties_df"], use_container_width=True)

            with tab_num:
                st.markdown("#### Kolmogorov-Smirnov (KS) Test & Distribution Divergence")
                st.dataframe(q_res["numerical_metrics"], use_container_width=True)

            with tab_cat:
                st.markdown("#### Total Variation Distance (TVD) & Category Coverage")
                st.dataframe(q_res["categorical_metrics"], use_container_width=True)

# =========================================================
# PAGE 8: PRIVACY ANALYSIS
# =========================================================
elif page == "Privacy Analysis":
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">🛡️ Empirical Privacy & Disclosure Risk Analysis</div>
        <p style="color: #475569; font-size: 14px;">
            Evaluates re-identification hazards through exact duplicate matching and Distance to Closest Record (DCR) analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    real_data = st.session_state.preprocessed_df if st.session_state.preprocessed_df is not None else st.session_state.raw_df
    synth_data = st.session_state.synthetic_df

    if real_data is None or synth_data is None:
        st.markdown("""
        <div class="empty-state">
            <h3>Prerequisites Incomplete</h3>
            <p>Generate synthetic records first before running privacy risk analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("🛡️ Execute Empirical Privacy Audit", use_container_width=True):
            with st.spinner("Computing Distance to Closest Record (DCR) and exact matches..."):
                try:
                    analyzer = PrivacyRiskAnalyzer(real_df=real_data, synthetic_df=synth_data)
                    priv_res = analyzer.analyze_privacy()
                    st.session_state.privacy_results = priv_res
                    gc.collect()
                    st.success("Privacy risk audit complete!")
                except Exception as e:
                    st.error(f"Privacy analysis error: {str(e)}")

        if st.session_state.privacy_results is not None:
            priv_res = st.session_state.privacy_results
            dcr_data = priv_res["distance_metrics"]
            match_data = priv_res["exact_matches"]

            st.markdown("---")
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("Overall Privacy Risk", priv_res["overall_privacy_risk"].split()[0])
            col_p2.metric("Exact Matches Found", f"{match_data['exact_matches_count']} ({match_data['match_rate_pct']}%)")
            col_p3.metric("Minimum DCR", f"{dcr_data['min_dcr']}")
            col_p4.metric("5th Percentile DCR", f"{dcr_data['p5_dcr']}")

            st.markdown(f"""
            <div style="background: #F0F8FF; border-left: 4px solid #008DDA; padding: 16px; border-radius: 8px; margin: 16px 0;">
                <b>Risk Assessment:</b> {dcr_data['risk_description']}
            </div>
            """, unsafe_allow_html=True)

            fig_dcr = plot_dcr_histogram(dcr_data["dcr_distribution"])
            st.plotly_chart(fig_dcr, use_container_width=True)

            st.warning(f"**Scientific & Legal Disclaimer:** {priv_res['disclaimer']}")

# =========================================================
# PAGE 9: ANALYTICS DASHBOARD
# =========================================================
elif page == "Analytics Dashboard":
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">📈 Interactive Research Visualization Dashboard</div>
        <p style="color: #475569; font-size: 14px;">
            Inspect distribution overlays, side-by-side categorical bars, correlation matrices, and 2D PCA manifold projections.
        </p>
    </div>
    """, unsafe_allow_html=True)

    real_data = st.session_state.preprocessed_df if st.session_state.preprocessed_df is not None else st.session_state.raw_df
    synth_data = st.session_state.synthetic_df

    if real_data is None or synth_data is None:
        st.markdown("""
        <div class="empty-state">
            <h3>No Synthetic Data Available</h3>
            <p>Train the CTGAN model and generate synthetic data to explore the interactive visual analytics.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        num_cols = [c for c in real_data.columns if c in synth_data.columns and pd.api.types.is_numeric_dtype(real_data[c]) and c not in st.session_state.discrete_columns]
        cat_cols = [c for c in real_data.columns if c in synth_data.columns and c in st.session_state.discrete_columns]

        tab_num, tab_cat, tab_corr, tab_pca = st.tabs([
            "📊 Numerical Distributions",
            "📊 Categorical Frequencies",
            "🔥 Correlation Heatmaps",
            "🌌 2D PCA Manifold"
        ])

        with tab_num:
            if num_cols:
                sel_num = st.selectbox("Select Numerical Feature to Compare", num_cols)
                fig_num = plot_numerical_distribution(real_data, synth_data, sel_num)
                st.plotly_chart(fig_num, use_container_width=True)
            else:
                st.info("No continuous numerical features detected.")

        with tab_cat:
            if cat_cols:
                sel_cat = st.selectbox("Select Categorical Feature to Compare", cat_cols)
                fig_cat = plot_categorical_distribution(real_data, synth_data, sel_cat)
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No categorical features detected.")

        with tab_corr:
            if len(num_cols) >= 2:
                fig_r, fig_s, fig_d = plot_correlation_heatmaps(real_data, synth_data, num_cols)
                c_c1, c_c2 = st.columns(2)
                with c_c1:
                    st.plotly_chart(fig_r, use_container_width=True)
                with c_c2:
                    st.plotly_chart(fig_s, use_container_width=True)
                st.plotly_chart(fig_d, use_container_width=True)
            else:
                st.info("At least 2 numerical features are required for correlation heatmaps.")

        with tab_pca:
            if len(num_cols) >= 2:
                fig_pca = plot_pca_projection(real_data, synth_data, num_cols)
                if fig_pca:
                    st.plotly_chart(fig_pca, use_container_width=True)
            else:
                st.info("PCA requires at least 2 numerical features.")

# =========================================================
# PAGE 10: DOWNLOAD REPORTS
# =========================================================
elif page == "Download Reports":
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">📑 Automated Research Evaluation & Audit Report</div>
        <p style="color: #475569; font-size: 14px;">
            Compile an executive-ready HTML evaluation report suitable for final-year engineering project documentation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    real_data = st.session_state.raw_df
    synth_data = st.session_state.synthetic_df

    if real_data is None or synth_data is None:
        st.markdown("""
        <div class="empty-state">
            <h3>No Evaluation Data Ready</h3>
            <p>Generate synthetic records and execute quality & privacy evaluations to compile the final report.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.session_state.quality_results is None:
            with st.spinner("Generating quality audit metrics for report..."):
                evaluator = DataQualityEvaluator(
                    real_df=real_data,
                    synthetic_df=synth_data,
                    discrete_columns=st.session_state.discrete_columns
                )
                st.session_state.quality_results = evaluator.generate_full_evaluation()

        if st.session_state.privacy_results is None:
            with st.spinner("Generating privacy audit metrics for report..."):
                analyzer = PrivacyRiskAnalyzer(real_df=real_data, synthetic_df=synth_data)
                st.session_state.privacy_results = analyzer.analyze_privacy()

        raw_stats = profile_dataset(real_data)
        gan_config = st.session_state.training_status or DEFAULT_GAN_CONFIG

        html_report = generate_html_report(
            dataset_name=st.session_state.dataset_name,
            raw_stats=raw_stats,
            gan_config=gan_config,
            quality_results=st.session_state.quality_results,
            privacy_results=st.session_state.privacy_results
        )

        st.success("Technical audit report compiled and ready for export!")

        st.download_button(
            label="📥 Download Technical HTML Audit Report",
            data=html_report,
            file_name=f"SynTabGAN_Technical_Report_{st.session_state.dataset_name.replace(' ', '_')}.html",
            mime="text/html",
            use_container_width=True
        )

        st.markdown("### 👁️ Report Preview")
        st.components.v1.html(html_report, height=650, scrolling=True)

# =========================================================
# PAGE 11: ABOUT PROJECT
# =========================================================
elif page == "About Project":
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">🎓 Academic & Engineering Project Details</div>
        <p style="color: #475569; font-size: 15px; line-height: 1.8;">
            <b>PROJECT TITLE:</b> Design and Development of a Working Prototype for Synthetic Tabular Data Generator 
            Using a Generative Adversarial Network for Tabular Data (CTGAN)
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_ab1, col_ab2 = st.columns(2)
    with col_ab1:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">📚 System Architecture & Core Stack</div>
            <ul style="color: #334155; font-size: 14px; line-height: 1.8;">
                <li><b>Generative Synthesizer:</b> <code>sdv.single_table.CTGANSynthesizer</code> (SDV 2.x)</li>
                <li><b>Deep Learning Framework:</b> PyTorch 2.x with WGAN-GP and PacGAN Discriminator</li>
                <li><b>Metadata Engine:</b> SDV <code>Metadata</code> Specification V2</li>
                <li><b>Fidelity Metrics:</b> SDMetrics Single Table Quality Reports, KS-2samp, Total Variation Distance</li>
                <li><b>Privacy Metrics:</b> Distance to Closest Record (DCR), Exact Match Rate</li>
                <li><b>User Interface:</b> Streamlit with custom Dark Navy & Light Blue Research theme</li>
                <li><b>Visualizations:</b> Interactive Plotly Graph Objects & Express</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_ab2:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">🛡️ Compliance & Privacy Notice</div>
            <p style="color: #475569; font-size: 14px; line-height: 1.7;">
                This application operates entirely in local isolated execution environments or secure sandboxed cloud workers. 
                Uploaded datasets are never transmitted to external third-party cloud APIs.
                Empirical privacy metrics (DCR) assess re-identification proximity and memorization heuristics.
            </p>
            <div style="background: #EBF4F6; border: 1px solid #BAE6FD; padding: 12px; border-radius: 8px; font-size: 12px; color: #0369A1;">
                Designed for final-year engineering project demonstration and public cloud deployment on Streamlit Community Cloud.
            </div>
        </div>
        """, unsafe_allow_html=True)
