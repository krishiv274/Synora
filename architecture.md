# Synora — System Architecture

> **Synora v2** — EV Charging Demand Prediction + Agentic Infrastructure Planning  
> Shenzhen, China · UrbanEV Dataset · 275 Traffic Analysis Zones

---

## 1. End-to-End System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SYNORA SYSTEM                                    │
│                                                                         │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐                 │
│   │ Raw Data │───▶│  ML Pipeline │───▶│  Streamlit   │                 │
│   │ UrbanEV  │    │  (Training)  │    │  Dashboard   │                 │
│   └──────────┘    └──────────────┘    └──────┬───────┘                 │
│                                              │                          │
│                                              ▼                          │
│                                   ┌──────────────────┐                 │
│                                   │  Agentic Planner │                 │
│                                   │  (LangGraph+RAG) │                 │
│                                   └──────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ML Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE  (Notebooks 01–05)                 │
└─────────────────────────────────────────────────────────────────────┘

  Raw UrbanEV CSVs
  (occupancy, volume,
   duration, e_price,          Notebook 01
   s_price, zone-info)  ──────────────────▶  EDA & Inspection
                                                    │
                                                    ▼
                                            Notebook 02
                                       Reshape: Wide ──▶ Long
                                       Merge all CSVs
                                       → merged_hourly_data.csv
                                                    │
                                                    ▼
                                            Notebook 03
                                       Feature Engineering
                                       ┌─────────────────────────────┐
                                       │ Temporal: hour, dow, month  │
                                       │ Cyclical: sin/cos encoding  │
                                       │ Lag: 1h, 3h, 6h, 12h,      │
                                       │       24h                   │
                                       │ Rolling: mean/std 6h–12h    │
                                       │ Spatial: charge_density     │
                                       │ Price: total_price          │
                                       │ → 30 features total         │
                                       └─────────────────────────────┘
                                       → final_featured_dataset.csv
                                                    │
                                                    ▼
                                            Notebook 04
                                       Hyperparameter Tuning
                                       GridSearchCV + TimeSeriesSplit
                                                    │
                                                    ▼
                                            Notebook 05
                                       Model Training
                                  ┌─────────┬────────────┬──────────┐
                                  │   RF    │  XGBoost   │  LGBM    │
                                  │  occ    │    occ     │   occ    │
                                  │  vol    │    vol     │   vol    │
                                  └────┬────┴─────┬──────┴────┬─────┘
                                       │          │           │
                                       └──────────┴───────────┘
                                                  │
                                                  ▼
                                        6 .pkl model files
                                        saved to models/
```

---

## 3. ML Model Architecture

```
                         INPUT (30 Features)
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ Random Forest│   │   XGBoost    │   │   LightGBM   │
  │              │   │              │   │              │
  │ 100 trees    │   │ Gradient     │   │ Histogram-   │
  │ Bagging      │   │ Boosting +   │   │ based GBDT   │
  │ ensemble     │   │ Regulariz.   │   │ Leaf-wise    │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                  │                  │
         ▼                  ▼                  ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  Occupancy  │    │  Occupancy  │    │  Occupancy  │
  │  Volume     │    │  Volume     │    │  Volume     │
  └─────────────┘    └─────────────┘    └─────────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                            ▼
                  Per-zone, per-hour predictions
               occupancy (%) + volume (kWh) output

  TRAIN SPLIT: Sep 2022 – Jan 2023
  TEST SPLIT:  Feb 2023 (time-based, no leakage)
```

---

## 4. Streamlit Dashboard Pages

```
  streamlit_app.py
  │
  ├── [1] Overview ──────────── KPIs, R²/MAE bars, hourly pattern
  ├── [2] Model Comparison ──── Radar chart, metric cards, bar groups
  ├── [3] Predictions Explorer ─ Time series, scatter, residuals, table
  ├── [4] Feature Importance ─── Top-N bar, cross-model comparison
  ├── [5] Zone Analysis ──────── Shenzhen map, MAE heatmap, rankings
  ├── [6] About ─────────────── Project overview, tech stack
  └── [7] 🤖 Agentic Planner ── LangGraph agent UI (NEW)
```

---

## 5. Agentic AI Layer — LangGraph StateGraph

```
  User Natural Language Query
  "Plan infrastructure for high-demand zones next weekend"
              │
              ▼
  ┌─────────────────────────────────────────────────────┐
  │              SYNORA LANGGRAPH AGENT                 │
  │                                                     │
  │  SynoraState (TypedDict — 12 fields flowing through)│
  │  ┌─────────────────────────────────────────────┐   │
  │  │ query · zone_ids · time_window · predictions│   │
  │  │ anomalies · rag_context · rag_sources       │   │
  │  │ recommendation · report                     │   │
  │  │ needs_human_review · approved · agent_trace │   │
  │  └─────────────────────────────────────────────┘   │
  │                                                     │
  │  ┌─────────────────┐                               │
  │  │ demand_forecaster│  Node 1                      │
  │  │                 │  ─ Parse zones + time window  │
  │  │  .pkl models    │  ─ Load trained pkl models    │
  │  │  (features: 30) │  ─ Predict occupancy + volume │
  │  │  (or fallback)  │  ─ Statistical fallback if    │
  │  └────────┬────────┘     LFS stubs detected         │
  │           │                                         │
  │           ▼                                         │
  │  ┌─────────────────┐                               │
  │  │ anomaly_detector│  Node 2                       │
  │  │                 │  ─ occ > 85% → ANOMALY        │
  │  │  Thresholds:    │  ─ vol > p90 → ANOMALY        │
  │  │  occ  > 85%     │  ─ surge > 40% → CRITICAL     │
  │  │  vol  > p90     │  ─ Assigns severity:          │
  │  │  surge> 40%     │    critical / high / medium   │
  │  └────────┬────────┘                               │
  │           │                                         │
  │           ▼                                         │
  │  ┌─────────────────┐     ┌──────────────────────┐  │
  │  │  rag_retriever  │────▶│   ChromaDB           │  │
  │  │                 │     │   Vector Store        │  │
  │  │  Node 3         │     │   data/vectorstore/   │  │
  │  │  ─ Embed query  │◀────│                      │  │
  │  │  ─ Retrieve     │     │  Documents:           │  │
  │  │    top-k docs   │     │  • 275 zone profiles  │  │
  │  │  ─ Zone-level   │     │  • 275 demand stats   │  │
  │  │    context      │     │  • 5 planning reports │  │
  │  └────────┬────────┘     │  • Model metrics      │  │
  │           │              │  • Feature importance  │  │
  │           │              │                        │  │
  │           │              │  Embeddings:           │  │
  │           │              │  all-MiniLM-L6-v2      │  │
  │           │              │  (local, no API key)   │  │
  │           │              └──────────────────────┘  │
  │           ▼                                         │
  │  ┌─────────────────┐                               │
  │  │ planning_agent  │  Node 4                       │
  │  │                 │  ─ Constructs rich prompt from│
  │  │  LLM:           │    predictions + anomalies +  │
  │  │  Claude         │    RAG context                │
  │  │  claude-sonnet  │  ─ Calls Anthropic / OpenAI  │
  │  │  -4-20250514    │  ─ Rule-based fallback if     │
  │  │  (or GPT-4o)    │    no API key set             │
  │  └────────┬────────┘                               │
  │           │                                         │
  │           ▼                                         │
  │  ┌─────────────────┐                               │
  │  │ report_generator│  Node 5                       │
  │  │                 │  ─ Structured JSON report     │
  │  │                 │  ─ summary_statistics         │
  │  │                 │  ─ anomalies list             │
  │  │                 │  ─ predictions_by_zone        │
  │  │                 │  ─ rag_sources_used           │
  │  │                 │  ─ model_info                 │
  │  └────────┬────────┘                               │
  │           │                                         │
  │           ▼                                         │
  │  ┌─────────────────┐                               │
  │  │human_review_gate│  Node 6 — Conditional         │
  │  │                 │                               │
  │  │  Triggers if:   │                               │
  │  │  • Surge > 40%  │                               │
  │  │  • Piles > 10   │                               │
  │  │  • Criticals>5  │                               │
  │  └──┬──────────┬───┘                               │
  │     │          │                                    │
  └─────┼──────────┼────────────────────────────────────┘
        │          │
        ▼          ▼
   [approved]  [needs_review]
        │          │
        │          ▼
        │    Streamlit approval widget
        │    ✅ Approve / ❌ Reject
        │          │
        └──────────┘
              │
              ▼
        Final Report
        JSON + Markdown
        Download button
```

---

## 6. RAG Pipeline Detail

```
  Natural Language Question
         │
         ▼
  SentenceTransformer
  all-MiniLM-L6-v2
  (384-dim embedding)
         │
         ▼
  ChromaDB Cosine
  Similarity Search
         │
         ▼
  Top-K Documents Retrieved
  ┌──────────────────────────────────────────────────┐
  │  [zone_profile_106]                              │
  │  Zone 106 — occ mean 49.2%, 96 piles, high      │
  │                                                  │
  │  [synthetic_report_000]                          │
  │  High-Demand Zone Infrastructure Report:         │
  │  Add 8–12 DC fast-charging piles…               │
  │                                                  │
  │  [zone_demand_stats_106]                         │
  │  p90 occ = 56%, p90 vol = 115 kWh               │
  │  High-risk flag: YES                             │
  └──────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────┐
  │  SYSTEM PROMPT                                   │
  │  You are Synora, EV infrastructure expert…       │
  ├──────────────────────────────────────────────────┤
  │  USER PROMPT                                     │
  │  Context: [retrieved docs]                       │
  │  Question: [user query]                          │
  │  Predictions: [zone-level occ/vol]               │
  │  Anomalies: [flagged zones]                      │
  └──────────────────────────────────────────────────┘
         │
         ▼
  Claude claude-sonnet-4-20250514 / GPT-4o
         │
         ▼
  Grounded Answer
  + Source Citations
  + Actionable Recommendations
```

---

## 7. ChromaDB Knowledge Base Structure

```
  data/vectorstore/          ← Persistent ChromaDB storage
  │
  └── Collection: synora_knowledge
      │
      ├── zone_profile_{id}        (275 docs)
      │   metadata: zone_id, region, metric_type="zone_profile",
      │             cluster, num_stations, total_piles,
      │             mean_occ, mean_vol, p90_occ, p90_vol
      │
      ├── zone_demand_stats_{id}   (275 docs)
      │   metadata: zone_id, region, metric_type="demand_stats",
      │             p90_occ, p90_vol
      │
      ├── synthetic_report_{000–004} (5 docs)
      │   metadata: metric_type="planning_report",
      │             cluster (high/medium/low), region
      │
      ├── model_metric_{model}_{target} (6 docs, if available)
      │   metadata: metric_type="model_metric"
      │
      └── feature_importance_{model}_{target} (6 docs, if available)
          metadata: metric_type="feature_importance"
```

---

## 8. Data Flow Summary

```
  ┌────────────────────────────────────────────────────────────┐
  │  INPUT                                                     │
  │  User: "Identify congested zones next weekend"             │
  └──────────────────────────┬─────────────────────────────────┘
                             │
             ┌───────────────▼──────────────────┐
             │         LangGraph Agent           │
             │                                  │
             │  1. Parse query → zones, window  │
             │  2. Predict demand per zone      │──▶ .pkl models
             │  3. Detect anomalies             │
             │  4. Retrieve RAG context         │──▶ ChromaDB
             │  5. Generate recommendation      │──▶ Claude API
             │  6. Format structured report     │
             │  7. Gate: human review?          │
             └───────────────┬──────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────────────┐
  │  OUTPUT (Streamlit Page)                                   │
  │  • Live step-by-step node trace                           │
  │  • Predicted demand heatmap (occupancy % by zone)         │
  │  • Volume chart (kWh by zone)                             │
  │  • Demand surge vs baseline chart                         │
  │  • Anomaly alert cards (CRITICAL / HIGH / MEDIUM)         │
  │  • RAG source accordion (retrieved documents)             │
  │  • Infrastructure recommendation (formatted markdown)     │
  │  • Human approval widget (if triggered)                   │
  │  • Download: JSON report + Markdown report                │
  └────────────────────────────────────────────────────────────┘
```

---

## 9. File Structure

```
Synora/
├── streamlit_app.py              ← Dashboard (UI + Agentic Planner)
├── requirements.txt              ← All dependencies
├── architecture.md               ← This file
├── README.md                     ← Project overview
├── report.tex                    ← LaTeX technical report
│
├── agent/                        ← Agentic AI layer (NEW)
│   ├── __init__.py
│   ├── state.py                  ← SynoraState TypedDict
│   ├── rag_engine.py             ← ChromaDB ingestion + query
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
│   │   └── final_featured_dataset.csv  ← 32 features, 275 zones
│   └── vectorstore/              ← ChromaDB persistent store (NEW)
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
    ├── feature_importance/       ← Per-model importance CSVs
    └── charts/                   ← 19 evaluation charts
```

---

## 10. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Optional* | Claude claude-sonnet-4-20250514 API key |
| `OPENAI_API_KEY` | Optional* | GPT-4o fallback API key |
| `MODEL_PROVIDER` | Optional | `"anthropic"` (default) or `"openai"` |

> *If neither key is set, the planning_agent falls back to rule-based recommendations. All other pipeline stages (predict, detect, retrieve, report) work without any API key.

---

## 11. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data** | pandas, NumPy | Data loading, processing |
| **ML Models** | scikit-learn, XGBoost, LightGBM | Demand prediction |
| **Agent Framework** | LangGraph ≥ 0.2 | Multi-node StateGraph orchestration |
| **LLM Chaining** | LangChain ≥ 0.3, langchain-anthropic | Prompt + RAG chain |
| **Vector DB** | ChromaDB ≥ 0.5 | Semantic document retrieval |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Local embeddings, no API key |
| **LLM** | Anthropic Claude claude-sonnet-4-20250514 | Planning recommendations |
| **UI** | Streamlit ≥ 1.31 | Interactive dashboard |
| **Visualization** | Plotly, Matplotlib, Seaborn | Charts and maps |
