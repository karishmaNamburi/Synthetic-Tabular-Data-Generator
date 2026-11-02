"""
Automated Evaluation and Audit Report Generation Module.
Exports executive-ready HTML and Markdown reports for project review.
"""
from typing import Dict, Any
from datetime import datetime
import pandas as pd


def generate_html_report(
    dataset_name: str,
    raw_stats: Dict[str, Any],
    gan_config: Dict[str, Any],
    quality_results: Dict[str, Any],
    privacy_results: Dict[str, Any]
) -> str:
    """
    Generates a beautifully styled, self-contained HTML research report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    overall_score = quality_results.get("overall_score", 0.0)
    grade = quality_results.get("grade", "N/A")
    verdict = quality_results.get("verdict", "")

    props_df = quality_results.get("sdmetrics", {}).get("properties_df", pd.DataFrame())
    props_html = props_df.to_html(index=False, classes="report-table") if not props_df.empty else "<p>No property data</p>"

    num_df = quality_results.get("numerical_metrics", pd.DataFrame())
    num_html = num_df.to_html(index=False, classes="report-table") if not num_df.empty else "<p>No numerical features analyzed</p>"

    cat_df = quality_results.get("categorical_metrics", pd.DataFrame())
    cat_html = cat_df.to_html(index=False, classes="report-table") if not cat_df.empty else "<p>No categorical features analyzed</p>"

    privacy_risk = privacy_results.get("overall_privacy_risk", "Unknown")
    exact_matches = privacy_results.get("exact_matches", {}).get("exact_matches_count", 0)
    min_dcr = privacy_results.get("distance_metrics", {}).get("min_dcr", 0.0)
    median_dcr = privacy_results.get("distance_metrics", {}).get("median_dcr", 0.0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Synthetic Tabular Data Generator - Technical Audit Report</title>
<style>
    body {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #F8FAFC;
        color: #0F172A;
        margin: 0;
        padding: 40px;
        line-height: 1.6;
    }}
    .container {{
        max-width: 1000px;
        margin: 0 auto;
        background: #FFFFFF;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
    }}
    .header {{
        border-bottom: 2px solid #0B192C;
        padding-bottom: 20px;
        margin-bottom: 30px;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }}
    .header h1 {{
        margin: 0;
        color: #0B192C;
        font-size: 24px;
    }}
    .header p {{
        margin: 5px 0 0 0;
        color: #64748B;
        font-size: 13px;
    }}
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 30px;
    }}
    .metric-card {{
        background: #F0F8FF;
        border: 1px solid #BAE6FD;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }}
    .metric-title {{
        font-size: 12px;
        font-weight: 600;
        color: #0369A1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .metric-val {{
        font-size: 24px;
        font-weight: 700;
        color: #0B192C;
        margin-top: 6px;
    }}
    h2 {{
        color: #0B192C;
        font-size: 18px;
        border-left: 4px solid #008DDA;
        padding-left: 10px;
        margin-top: 30px;
        margin-bottom: 15px;
    }}
    .report-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0 25px 0;
        font-size: 13px;
    }}
    .report-table th {{
        background-color: #0B192C;
        color: #FFFFFF;
        text-align: left;
        padding: 10px 12px;
        font-weight: 600;
    }}
    .report-table td {{
        padding: 8px 12px;
        border-bottom: 1px solid #E2E8F0;
    }}
    .report-table tr:nth-child(even) {{
        background-color: #F8FAFC;
    }}
    .disclaimer-box {{
        background: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 8px;
        padding: 16px;
        margin-top: 30px;
        font-size: 13px;
        color: #92400E;
    }}
    .footer {{
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        color: #94A3B8;
        font-size: 12px;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1>Synthetic Tabular Data Generator - Technical Audit Report</h1>
            <p>Project Demonstration & Academic Evaluation | Generative Adversarial Networks (CTGAN)</p>
        </div>
        <p>Generated: <b>{timestamp}</b></p>
    </div>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">Quality Score</div>
            <div class="metric-val">{overall_score}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Fidelity Grade</div>
            <div class="metric-val">{grade.split()[0]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Exact Matches</div>
            <div class="metric-val">{exact_matches}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Privacy Risk</div>
            <div class="metric-val" style="font-size: 18px; margin-top: 10px;">{privacy_risk.split()[0]}</div>
        </div>
    </div>

    <h2>1. Executive Summary & Verdict</h2>
    <p><b>Dataset Evaluated:</b> {dataset_name}</p>
    <p><b>Model Verdict:</b> {verdict}</p>

    <h2>2. Dataset & CTGAN Training Configuration</h2>
    <table class="report-table">
        <tr><th>Parameter</th><th>Value</th><th>Parameter</th><th>Value</th></tr>
        <tr><td>Original Records</td><td>{raw_stats.get('rows', 'N/A')}</td><td>Training Epochs</td><td>{gan_config.get('epochs', 'N/A')}</td></tr>
        <tr><td>Feature Count</td><td>{raw_stats.get('columns', 'N/A')}</td><td>Batch Size</td><td>{gan_config.get('batch_size', 'N/A')}</td></tr>
        <tr><td>Missing Values (Cleaned)</td><td>{raw_stats.get('missing_cells', 'N/A')}</td><td>Pac Size</td><td>{gan_config.get('pac', 'N/A')}</td></tr>
        <tr><td>Training Execution Time</td><td>{gan_config.get('training_time', 'N/A')}</td><td>Compute Device</td><td>{gan_config.get('device', 'CPU')}</td></tr>
    </table>

    <h2>3. SDMetrics Single Table Quality Dimensions</h2>
    {props_html}

    <h2>4. Numerical Feature Kolmogorov-Smirnov (KS) Divergence</h2>
    {num_html}

    <h2>5. Categorical Feature Total Variation Distance (TVD)</h2>
    {cat_html}

    <h2>6. Empirical Privacy & Disclosure Risk Audit</h2>
    <table class="report-table">
        <tr><th>Metric</th><th>Observation</th><th>Interpretation</th></tr>
        <tr><td>Exact Matches Count</td><td>{exact_matches}</td><td>Zero matches indicate absence of direct record replication.</td></tr>
        <tr><td>Minimum Distance to Closest Record (DCR)</td><td>{min_dcr}</td><td>Distance from most similar synthetic sample to real training point.</td></tr>
        <tr><td>Median DCR</td><td>{median_dcr}</td><td>Typical separation between synthetic points and training data manifold.</td></tr>
    </table>

    <div class="disclaimer-box">
        <b>Ethical & Scientific Disclaimer:</b> {privacy_results.get('disclaimer', '')}
    </div>

    <div class="footer">
        Synthetic Tabular Data Generator using CTGAN &bull; Final-Year Engineering Demonstration &bull; Powered by PyTorch & SDV
    </div>
</div>
</body>
</html>
"""
    return html
