# Synora v2
### Agentic EV Charging Demand Prediction & Infrastructure Planning
> Shenzhen, China · UrbanEV Dataset · 275 Traffic Analysis Zones

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-6C63FF)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5%2B-00C9A7)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What is Synora?

**Synora** is a two-layer AI system for EV charging infrastructure planning:

| Layer | What it does |
|---|---|
| **ML Prediction** | 3 ensemble models (RF, XGBoost, LGBM) forecast occupancy (%) and volume (kWh) per zone per hour — realistic R² ~ 0.98 for occupancy, ~ 0.94 for volume |
| **Agentic Planner** | LangGraph multi-node agent combines ML predictions, anomaly detection, RAG over ChromaDB, and Claude claude-sonnet-4-20250514 to generate grounded infrastructure recommendations |

---

## Architecture

```
  User Query
      │
      ▼
  ┌─────────────────── LangGraph Agent ──────────────────────┐
  │                                                          │
  │  [1] demand_forecaster ──▶ Load .pkl → predict per zone │
  │       │                                                  │
  │  [2] anomaly_detector  ──▶ Flag occ>85% / surge>40%     │
  │       │                                                  │
  │  [3] rag_retriever     ──▶ Query ChromaDB               │
  │       │                    (275 zone profiles + reports) │
  │  [4] planning_agent    ──▶ Claude claude-sonnet-4-20250514 → recommendation │
  │       │                                                  │
  │  [5] report_generator  ──▶ Structured JSON + Markdown   │
  │       │                                                  │
  │  [6] human_review_gate ──▶ Auto-approve or flag for      │
  │                             human sign-off               │
  └──────────────────────────────────────────────────────────┘
      │
      ▼
  Synora Dashboard (Streamlit)
  • Live agent trace   • Demand heatmaps
  • Anomaly alerts     • RAG source viewer
  • Recommendation     • JSON/MD download
```

> Full architecture diagrams with ASCII art: see [`architecture.md`](architecture.md)

---

## End-semester submission checklist (rubric alignment)

This table maps the **GenAI & Agentic AI** end-sem evaluation components to artifacts in this repository.

| Component (weight) | Where it is satisfied |
|--------------------|------------------------|
| **Technical implementation (35%)** | `agent/` — LangGraph `StateGraph` (`graph.py`), explicit `SynoraState` (`state.py`), six nodes (`nodes.py`), Chroma RAG (`rag_engine.py`, `rag_pipeline.py`). Structured JSON report fields: `charging_demand_summary`, `high_load_zone_ids` / `high_load_locations`, `charger_placement_priorities`, `scheduling_insights`, `grounding_and_retrieval`. ML layer supports demand signals; **agentic** path is mandatory for grading. |
| **GitHub & code quality (15%)** | Modular `agent/` vs UI in `streamlit_app.py`, `requirements.txt`, `.env.example`, `LICENSE`, notebooks under `notebooks/`. |
| **Hosted demo (15%)** | Deploy with [Streamlit Community Cloud](https://streamlit.io/cloud) or a Hugging Face Space (see **Hosted demo** below). |
| **Project report — LaTeX (20%)** | `report.tex` — compile with `pdflatex report.tex` (twice if TOC updates). |
| **Project video (15%)** | Record a **~5 min** walkthrough: ML overview → **Agentic Planner** query → live node trace → RAG sources → structured expander → export JSON. |

**Milestone 2 (agentic) functional coverage**

| Requirement | Implementation |
|-------------|----------------|
| Analyse demand for high-load locations | `demand_forecaster`, `charging_demand_summary`, ranked `high_load_locations` |
| Retrieve planning guidelines | `rag_retriever` → ChromaDB collection `synora_knowledge`; graceful degradation if retrieval fails |
| Charger placement recommendations | `planning_agent` + `charger_placement_priorities` in JSON report |
| Scheduling optimization insights | `scheduling_insights` dict + **Scheduling & operations** section in LLM / rule-based markdown |
| LangGraph workflow & state | `graph.py`, `SynoraState` |
| Structured output / evaluation | Streamlit expander **Structured planning outputs**, downloadable JSON/MD |

---

## Technology Stack

| Component | Technology |
|---|---|
| **ML Models** | Random Forest, XGBoost, LightGBM (scikit-learn) |
| **Agent Framework** | LangGraph ≥ 0.2, LangChain ≥ 0.3 |
| **Vector Database** | ChromaDB ≥ 0.5 (persisted at `data/vectorstore/`) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, no API key) |
| **LLM** | Default: Groq `llama-3.3-70b-versatile` (free tier); optional Anthropic Claude `claude-sonnet-4-20250514` or OpenAI `gpt-4o` |
| **Dashboard** | Streamlit ≥ 1.31 |
| **Visualisation** | Plotly, Matplotlib, Seaborn |
| **Language** | Python 3.9+ |

---

## Dataset — UrbanEV (Shenzhen, China)

| Property | Value |
|---|---|
| Time Range | Sep 2022 – Feb 2023 (6 months, hourly) |
| Spatial Units | 275 Traffic Analysis Zones (TAZs) |
| Charging Stations | 1,362 |
| Charging Piles | 17,532 |
| Prediction Targets | Occupancy (%) · Volume (kWh) |

**Source files used:**

| File | Description |
|---|---|
| `occupancy.csv` | Hourly station utilisation rate per zone (%) |
| `volume.csv` | Total energy dispensed per zone per hour (kWh) |
| `duration.csv` | Aggregated charging durations |
| `e_price.csv` | Electricity price (CNY/kWh) |
| `s_price.csv` | Service fee (CNY/kWh) |
| `zone-information.csv` | Zone coordinates, area, perimeter, pile counts |
| `station_information.csv` | Station-level coordinates and pile counts |

---

## Model Results

### Occupancy Prediction

| Model | MAE | RMSE | R² | MAPE (%) |
|---|---|---|---|---|
| **Random Forest** | 1.2502 | 2.5798 | **0.9837** | 1.22 |
| LightGBM | 1.2467 | 2.5725 | 0.9834 | 1.34 |
| XGBoost | 1.2368 | 2.5522 | 0.9827 | 1.62 |

### Volume Prediction

| Model | MAE | RMSE | R² | MAPE (%) |
|---|---|---|---|---|
| **Random Forest** | 54.27 | 202.81 | **0.9437** | 43.89 |
| LightGBM | 53.21 | 198.85 | 0.9418 | 48.78 |
| XGBoost | 53.76 | 200.93 | 0.9428 | 45.87 |

> All three models achieve **a realistic R² > 0.94** on both targets, entirely free of target data leakage.

---

## Agentic Planner — Node Details

| Node | Trigger condition | Output |
|---|---|---|
| `demand_forecaster` | Always runs | Predictions dict (occupancy % + volume kWh per zone) |
| `anomaly_detector` | Always runs | Flagged zones with severity (critical / high / medium) |
| `rag_retriever` | Always runs | Top-k ChromaDB docs (zone profiles + planning reports) |
| `planning_agent` | Always runs | LLM recommendation markdown (rule-based fallback if no API key) |
| `report_generator` | Always runs | Structured JSON report with unique report ID |
| `human_review_gate` | surge > 40% OR piles > 10 OR criticals > 5 | `needs_human_review` flag → Streamlit approval widget |

### Anomaly Detection Thresholds

| Condition | Threshold | Severity |
|---|---|---|
| Predicted occupancy | > 95% | critical |
| Predicted occupancy | > 85% | high |
| Demand surge vs baseline | > 60% | critical |
| Demand surge vs baseline | > 40% | high |
| Volume vs zone 90th percentile | exceeded | medium |

### ChromaDB Knowledge Base

```
data/vectorstore/  ←  Persistent ChromaDB store
└── Collection: synora_knowledge
    ├── zone_profile_{id}        275 docs  — spatial + demand features per zone
    ├── zone_demand_stats_{id}   275 docs  — mean, std, p90 occ + vol per zone
    ├── synthetic_report_{0–4}     5 docs  — infrastructure planning templates
    ├── model_metric_{model}_{t}   6 docs  — MAE, RMSE, R², MAPE per model
    └── feature_importance_{…}     6 docs  — top features per model/target
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| **📊 Overview** | KPI cards, R²/MAE bar charts, metrics table, hourly demand patterns, all-models overlay |
| **📈 Model Comparison** | Model metric cards, grouped bars, radar chart |
| **🔍 Predictions Explorer** | Time series, scatter plot, error histogram, residual plot, data table |
| **🎯 Feature Importance** | Top-N bars, cross-model normalised comparison, importance table |
| **🗺️ Zone Analysis** | Interactive Shenzhen map, MAE distribution, demand vs error scatter, zone rankings |
| **ℹ️ About** | Project overview, tech stack, dataset citation |
| **🤖 Agentic Planner** | Query input, live node trace, demand heatmaps, anomaly alerts, RAG source viewer, structured rubric JSON (demand summary, high-load IDs, scheduling), recommendation, human approval widget, JSON/MD export |

---

## Feature Engineering (30 Features)

| Category | Features |
|---|---|
| Spatial | longitude, latitude, area, perimeter, num_stations, total_piles, mean_station_lat, mean_station_lon |
| Temporal | hour, day_of_week, month, day_of_month, is_weekend |
| Cyclical (sin/cos) | hour_sin, hour_cos, dow_sin, dow_cos |
| Lag — Occupancy | occ_lag_1h, occ_lag_3h, occ_lag_6h, occ_lag_12h, occ_lag_24h, occ_lag_168h |
| Lag — Volume | vol_lag_24h |
| Rolling Mean | occ_rmean_6h, occ_rmean_12h, occ_rmean_24h |
| Rolling Std | occ_rstd_24h |
| Differencing | occ_diff_1h |
| Price | total_price |
| Spatial derived | charge_density removed (leakage prevention) |

**Train / Test Split:** Sep 2022–Jan 2023 train · Feb 2023 test (time-based, no leakage)

---

## Project Structure

```
Synora/
├── streamlit_app.py              ← Dashboard (Streamlit UI + Agentic Planner)
├── requirements.txt              ← All Python dependencies
├── architecture.md               ← Full architecture diagrams
├── README.md                     ← This file
├── .env.example                  ← API key template (copy to .env locally)
├── report.tex                    ← LaTeX technical report
│
├── agent/                        ← Agentic AI layer
│   ├── __init__.py
│   ├── state.py                  ← SynoraState TypedDict (12 fields)
│   ├── rag_engine.py             ← ChromaDB ingestion + semantic query
│   ├── rag_pipeline.py           ← LangChain RAG chain
│   ├── nodes.py                  ← 6 LangGraph node functions
│   └── graph.py                  ← StateGraph wiring + streaming
│
├── data/
│   ├── raw/
│   │   ├── charge_1hour/         ← 6 raw hourly CSVs
│   │   ├── zone-information.csv
│   │   └── station_information.csv
│   ├── processed/
│   │   ├── merged_hourly_data.csv
│   │   └── final_featured_dataset.csv  ← 30 features, ~1.5M rows
│   └── vectorstore/              ← ChromaDB persistent store
│
├── models/
│   ├── randomforest_occupancy.pkl
│   ├── randomforest_volume.pkl
│   ├── xgboost_occupancy.pkl
│   ├── xgboost_volume.pkl
│   ├── lightgbm_occupancy.pkl
│   └── lightgbm_volume.pkl
│
├── notebooks/
│   ├── 01 data_inspection.ipynb
│   ├── 02 reshape_dataset.ipynb
│   ├── 03 feature_engineering.ipynb
│   ├── 04 hyperparameter_tuning.ipynb
│   ├── 05 model_training.ipynb
│   └── 06 visualization.ipynb
│
└── results/
    ├── metrics/model_metrics.csv
    ├── predictions/test_predictions.csv
    ├── feature_importance/
    └── charts/                    ← 19 evaluation charts
```

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/krishiv274/Synora.git
cd Synora

# 2. (Optional) Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install all dependencies
python3 -m pip install -r requirements.txt

# 4. API keys (optional — rule-based planner works without any key)
#    Copy .env.example → .env, fill keys, then export, or use:
export GROQ_API_KEY="gsk_..."           # recommended free tier
# export ANTHROPIC_API_KEY="sk-ant-..."
# export OPENAI_API_KEY="sk-..."
# export MODEL_PROVIDER="groq"        # groq | anthropic | openai

# 5. Run the dashboard
python3 -m streamlit run streamlit_app.py
```

Then open **http://localhost:8501** and open **🤖 Agentic Planner** from the sidebar (it appears first in the navigation list).

### Environment Variables

See [`.env.example`](.env.example) for a template. **Do not commit `.env`** (it is gitignored).

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Optional | — | Groq API key (free); enables Llama 3.3 70B in the planner |
| `ANTHROPIC_API_KEY` | Optional | — | Claude claude-sonnet-4-20250514 |
| `OPENAI_API_KEY` | Optional | — | GPT-4o when `MODEL_PROVIDER=openai` |
| `MODEL_PROVIDER` | Optional | `groq` | `groq`, `anthropic`, or `openai` |

> **Without any API key:** the agent uses a deterministic rule-based planner. All other nodes (predict, detect, retrieve, report) work fully without any API key.

### Hosted demo (Streamlit Community Cloud)

1. Push this repo to GitHub (include `requirements.txt`, `streamlit_app.py`, `agent/`, `models/`, `data/vectorstore/` as your policy allows — large CSVs may stay out; the agent falls back if models/data are stubs).
2. On [share.streamlit.io](https://share.streamlit.io), **New app** → select the repo → Main file: `streamlit_app.py`.
3. Under **Secrets**, add TOML, for example:
   ```toml
   GROQ_API_KEY = "gsk_..."
   MODEL_PROVIDER = "groq"
   ```
4. Redeploy. Put the public URL in your report cover sheet and video description.

**Hugging Face Spaces:** create a **Docker** or **Gradio/Streamlit** Space, `COPY` the repo, `pip install -r requirements.txt`, set the same env vars under **Settings → Repository secrets**.

---

## Notebook Pipeline

| # | Notebook | Description |
|---|---|---|
| 1 | `01 data_inspection.ipynb` | EDA — 7 visualisations (daily trends, hourly profiles, distributions, correlation heatmap, zone analysis) |
| 2 | `02 reshape_dataset.ipynb` | Melt wide CSVs → long, merge zone + station metadata |
| 3 | `03 feature_engineering.ipynb` | Engineer 30 features → `final_featured_dataset.csv` (No Target Leakage) |
| 4 | `04 hyperparameter_tuning.ipynb` | GridSearchCV + TimeSeriesSplit (no look-ahead bias) |
| 5 | `05 model_training.ipynb` | Train 6 models, evaluate on Feb 2023 test set |
| 6 | `06 visualization.ipynb` | 19 charts: scatter, residuals, feature importance, zone RMSE |

---

## References

1. Li, H., et al. (2025). *UrbanEV: An open benchmark dataset for urban electric vehicle charging demand prediction.* Scientific Data.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
3. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
4. Breiman, L. (2001). *Random Forests.* Machine Learning.
5. LangChain AI (2023). *LangGraph.* https://langchain-ai.github.io/langgraph/
6. Chroma (2023). *ChromaDB.* https://www.trychroma.com
7. Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.
8. Anthropic (2024). *Claude claude-sonnet-4-20250514.* https://www.anthropic.com

---

> **Location:** Shenzhen, China · **Dataset:** UrbanEV · **Period:** Sep 2022 – Feb 2023
