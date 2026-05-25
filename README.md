# 🛡️ Insurance EDA Engine
### Production-Grade Exploratory Data Analysis | Sam Adesina

> **Simulated client: ShieldGuard Insurance Group** — Following a quarter where the loss ratio exceeded 85%, this project delivers a board-ready statistical analysis identifying which claim types and customer risk grades drive the highest losses.

---

## What This Project Does

This is a **fully engineered EDA system** — not a notebook with charts. It is a modular Python application with custom classes for statistical analysis, anomaly detection, and HTML report generation, built on top of clean insurance claims data processed by a separate ETL pipeline.

The system analyses **20,004 insurance claims across 42 columns**, running ANOVA, Kruskal-Wallis, Pearson correlation ranking, exponential smoothing forecasts, and fraud pattern profiling — all compiled into a self-contained HTML report.

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Data Input** | `processed-data.csv` from insurance ETL pipeline |
| **Statistical Analysis** | `scipy.stats` — ANOVA, Kruskal-Wallis |
| **Forecasting** | `statsmodels` — exponential smoothing |
| **Anomaly Detection** | IQR + Z-score consensus, fraud pattern profiling |
| **Reporting** | Custom `HTMLReporter` → `reports/insurance_eda_report.html` |
| **Visualisation** | `matplotlib`, `seaborn` — 6 chart types |
| **Notebook** | Jupyter — interactive analysis layer |
| **Environment** | Python 3.11, venv |

---

## Project Structure

```
insurance-eda/
├── data/
│   └── processed-data.csv          # Clean input (20,004 rows, from ETL pipeline)
├── notebooks/
│   └── insurance_deep_dive.ipynb   # Interactive analysis — 6 charts
├── reports/
│   ├── insurance_eda_report.html   # Board-ready HTML report
│   ├── anomalies.csv               # Flagged anomalous claims
│   └── *.png                       # Saved chart outputs
├── src/
│   ├── eda_engine.py               # InsuranceEDAEngine + statistical methods
│   └── anomaly_detector.py         # AnomalyDetector + fraud pattern profiling
├── config.py
└── run.py                          # Single entry point
```

---

## Core Components

### `InsuranceEDAEngine`
Full EDA class with domain-specific statistical methods:

- `load()` / `profile()` — dataset overview, descriptive statistics, null/duplicate checks
- `group_analysis()` — fraud rate by `claim_type`, `risk_grade`, `policy_type`
- `correlation()` — Pearson correlation matrix across all numeric features
- `anova_test(group_col, value_col)` — one-way ANOVA via `scipy.stats.f_oneway`
- `kruskal_test(group_col, value_col)` — non-parametric alternative for skewed claim data
- `feature_importance(target_col)` — Pearson correlation ranking of all features against a target
- `forecast_next_period(col)` — exponential smoothing via `statsmodels`
- `time_trends()` — month-over-month claim volume and fraud rate trends
- `report()` — structured text report with next steps

### `AnomalyDetector`
Statistical anomaly detection using IQR + Z-score consensus:

- `run()` — flags confirmed anomalies using both methods independently; a record is confirmed only if flagged by both (reduces false positives)
- `flag_fraud_patterns()` — profiles fraudulent vs legitimate claims and isolates **evasive fraud**: confirmed fraudulent claims with normal amounts that bypass automated screening

### `HTMLReporter`
Generates a self-contained HTML report with:
- Dataset overview and descriptive statistics
- ANOVA and Kruskal-Wallis results with p-value interpretation
- Feature importance table ranked by Pearson r
- Exponential smoothing forecast with business interpretation
- Fraud pattern summary by claim type and risk grade
- Anomaly summary table
- Saved to `reports/insurance_eda_report.html` — opens in any browser, no dependencies

---

## Key Analytical Results

| Analysis | Method | Output |
|---|---|---|
| Claim amount by risk grade | One-way ANOVA + Kruskal-Wallis | F-statistic, p-value, conclusion |
| Feature importance for `amount_approved` | Pearson correlation ranking | Ranked feature table |
| Next period claim volume | Exponential smoothing | Forecast value + % change |
| Fraud pattern profile | IQR bounds + group statistics | Evasive vs high-value fraud split |
| Anomaly detection | IQR + Z-score consensus | Confirmed anomaly rows |

---

## Notebook Charts (6 Required)

| Chart | Type | File |
|---|---|---|
| Claim amount distribution | Histogram + boxplot | `dist_amount_claimed.png` |
| Claim amount by risk grade | Group comparison boxplot | `group_comparison_risk_grade.png` |
| Feature correlations | Heatmap | `correlation_heatmap.png` |
| Monthly claim volume | Time trend line | `claim_volume_forecast.png` |
| Fraudulent vs legitimate | Anomaly scatter | `anomaly_scatter.png` |
| Feature importance | Ranked bar chart | `feature_importance.png` |

---

## How to Run

```bash
# 1. Clone and set up environment
git clone https://github.com/samuadesina/insurance-eda.git
cd insurance-eda
python -m venv insurance-venv
source insurance-venv/bin/activate
pip install -r requirements.txt

# 2. Add processed-data.csv to data/
# (output from https://github.com/samuadesina/insurance-etl)

# 3. Run full analysis
python run.py

# 4. Open the report
open reports/insurance_eda_report.html
```

---

## Part of a Larger Portfolio

This project is the **analysis layer** that sits on top of the insurance ETL pipeline. Together they form a complete data engineering + analysis stack built end-to-end.

| Project | Description |
|---|---|
| [insurance-etl](https://github.com/samuadesina/insurance-etl) | ETL pipeline — extract, validate, transform (PostgreSQL / Supabase) |
| **insurance-eda** | **EDA engine — statistical analysis, fraud detection, HTML report** |
| [banking-etl](https://github.com/samuadesina/banking-etl) | Banking — transactions, fraud alerts, loan risk (PostgreSQL) |

---

## Skills Demonstrated

`Python` · `pandas` · `scipy` · `statsmodels` · `matplotlib` · `seaborn` · `statistical hypothesis testing` · `ANOVA` · `Kruskal-Wallis` · `exponential smoothing` · `anomaly detection` · `IQR` · `Z-score` · `Pearson correlation` · `HTML report generation` · `Jupyter` · `OOP`

---

*Built by [GitHub](https://github.com/samuadesina) · ([LinkedIn](https://www.linkedin.com/in/samuadesina/)) — MSc Data Science | Open to data engineering and analyst roles.*
