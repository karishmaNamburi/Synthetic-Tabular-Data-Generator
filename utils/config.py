"""
Configuration, constants, theme palette, and resource limits.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

SAMPLE_DATASETS = {
    "Employee Attrition (HR Analytics)": SAMPLE_DATA_DIR / "employee_attrition.csv",
    "Loan Risk Assessment (FinTech / Credit)": SAMPLE_DATA_DIR / "loan_risk_assessment.csv",
    "Patient Health Records (Healthcare Diagnostics)": SAMPLE_DATA_DIR / "patient_health_records.csv"
}

# Streamlit Community Cloud Resource Constraints
MAX_CLOUD_ROWS = 5000       # Maximum dataset rows allowed before warning/downsampling
SAFE_SAMPLE_ROWS = 3000     # Suggested downsample size to prevent Cloud OOM kills
MAX_CLOUD_EPOCHS = 100      # Max epochs allowed in UI
DEFAULT_CLOUD_EPOCHS = 35   # Optimal default epochs for cloud compute
MAX_UPLOAD_MB = 10          # Maximum file upload limit

THEME = {
    "dark_navy": "#0B192C",
    "navy_card": "#1E3E62",
    "accent_blue": "#008DDA",
    "light_blue": "#41B06E",
    "ice_blue": "#F0F8FF",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFC",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "border": "#E2E8F0",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#3B82F6"
}

DEFAULT_GAN_CONFIG = {
    "epochs": DEFAULT_CLOUD_EPOCHS,
    "batch_size": 100,
    "pac": 10,
    "embedding_dim": 128,
    "generator_dim": (256, 256),
    "discriminator_dim": (256, 256),
    "generator_lr": 2e-4,
    "discriminator_lr": 2e-4,
    "generator_decay": 1e-6,
    "discriminator_decay": 1e-6,
    "verbose": False
}

LOGO_SVG = """
<svg width="42" height="42" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="48" height="48" rx="12" fill="url(#paint0_linear)"/>
  <circle cx="15" cy="15" r="4" fill="#00D2FF"/>
  <circle cx="33" cy="15" r="4" fill="#00D2FF"/>
  <circle cx="24" cy="33" r="4.5" fill="#FFFFFF"/>
  <path d="M15 15L24 33L33 15" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" stroke-opacity="0.85"/>
  <path d="M15 15H33" stroke="#00D2FF" stroke-width="2" stroke-linecap="round" stroke-dasharray="2 3"/>
  <circle cx="24" cy="23" r="2.5" fill="#38BDF8"/>
  <defs>
    <linearGradient id="paint0_linear" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
      <stop stop-color="#0B192C"/>
      <stop offset="1" stop-color="#1E3E62"/>
    </linearGradient>
  </defs>
</svg>
"""
