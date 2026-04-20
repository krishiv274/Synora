# Synora
### Agentic EV Charging Demand Prediction & Infrastructure Planning

> Shenzhen, China · UrbanEV Dataset · 275 Traffic Analysis Zones  

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-6C63FF)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5%2B-00C9A7)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🚀 Demo

🔗 **Live App:** [Check live app demo](https://synora.streamlit.app/)

---

## 📌 Overview

**Synora** is a two-layer AI system designed for **EV charging infrastructure planning**:

- **ML Prediction Layer**  
  Ensemble models (Random Forest, XGBoost, LightGBM) predict:
  - Charging occupancy (%)
  - Energy demand (kWh)

- **Agentic Planning Layer**  
  A LangGraph-based multi-node system that:
  - Detects anomalies  
  - Retrieves contextual data (RAG via ChromaDB)  
  - Generates infrastructure recommendations using LLMs  

---

## 🧠 System Architecture

```mermaid
flowchart TD
    A[User Query] --> B[LangGraph Agent]

    B --> C[demand_forecaster]
    C --> D[anomaly_detector]
    D --> E[rag_retriever]
    E --> F[planning_agent]
    F --> G[report_generator]
    G --> H[human_review_gate]

    H --> I[Streamlit Dashboard]

    subgraph Dashboard
        I --> I1[Demand Heatmaps]
        I --> I2[Anomaly Alerts]
        I --> I3[RAG Viewer]
        I --> I4[Recommendations]
        I --> I5[Exports JSON/MD]
    end
````

---

## 🛠️ Tech Stack

| Layer           | Technology                                        |
| --------------- | ------------------------------------------------- |
| ML Models       | Random Forest, XGBoost, LightGBM                  |
| Agent Framework | LangGraph, LangChain                              |
| Vector DB       | ChromaDB                                          |
| Embeddings      | sentence-transformers (MiniLM)                    |
| LLM             | Groq (Llama 3.3 70B), optional OpenAI / Anthropic |
| Dashboard       | Streamlit                                         |
| Visualisation   | Plotly, Matplotlib                                |
| Language        | Python 3.9+                                       |

---

## 📊 Dataset — UrbanEV

| Property       | Value               |
| -------------- | ------------------- |
| Location       | Shenzhen, China     |
| Duration       | Sep 2022 – Feb 2023 |
| Zones          | 275 TAZs            |
| Stations       | 1,362               |
| Charging Piles | 17,532              |

### Key Files

* `occupancy.csv`
* `volume.csv`
* `duration.csv`
* `zone-information.csv`
* `station_information.csv`

---

## 📈 Model Performance

### Occupancy Prediction

| Model         | R²         |
| ------------- | ---------- |
| Random Forest | **0.9837** |
| LightGBM      | 0.9834     |
| XGBoost       | 0.9827     |

### Volume Prediction

| Model         | R²         |
| ------------- | ---------- |
| Random Forest | **0.9437** |
| XGBoost       | 0.9428     |
| LightGBM      | 0.9418     |

---

## 🤖 Agentic Planner

### Workflow Nodes

| Node              | Purpose                        |
| ----------------- | ------------------------------ |
| demand_forecaster | Predicts demand per zone       |
| anomaly_detector  | Flags high-load zones          |
| rag_retriever     | Retrieves contextual documents |
| planning_agent    | Generates recommendations      |
| report_generator  | Outputs structured reports     |
| human_review_gate | Flags critical decisions       |

---

## ⚠️ Anomaly Detection Rules

| Condition    | Threshold        |
| ------------ | ---------------- |
| Occupancy    | > 95% (critical) |
| Occupancy    | > 85% (high)     |
| Demand Surge | > 60% (critical) |
| Demand Surge | > 40% (high)     |

---

## 🧩 Feature Engineering

**30 engineered features including:**

* Spatial: coordinates, station density
* Temporal: hour, weekday, month
* Cyclical: sin/cos encoding
* Lag features: up to 168h
* Rolling statistics

---

## 🖥️ Dashboard Features

* 📊 Model performance comparison
* 🗺️ Interactive zone analysis
* 📉 Prediction visualisations
* 🤖 Agentic planner interface
* 📄 Exportable reports

---

## 📁 Project Structure

```bash
Synora/
├── streamlit_app.py
├── agent/
├── data/
├── models/
├── notebooks/
├── results/
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/krishiv274/Synora.git
cd Synora

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

export GROQ_API_KEY="your_key"

streamlit run streamlit_app.py
```

---

## 🔐 Environment Variables

| Variable       | Description               |
| -------------- | ------------------------- |
| GROQ_API_KEY   | Enables Llama 3.3         |
| MODEL_PROVIDER | groq / openai / anthropic |

---

## 📓 Pipeline

| # | Notebook | Description |
|---|---|---|
| 1 | `01 data_inspection.ipynb` | EDA — 7 visualisations (daily trends, hourly profiles, distributions, correlation heatmap, zone analysis) |
| 2 | `02 reshape_dataset.ipynb` | Melt wide CSVs → long, merge zone + station metadata |
| 3 | `03 feature_engineering.ipynb` | Engineer 30 features → `final_featured_dataset.csv` (No Target Leakage) |
| 4 | `04 hyperparameter_tuning.ipynb` | GridSearchCV + TimeSeriesSplit (no look-ahead bias) |
| 5 | `05 model_training.ipynb` | Train 6 models, evaluate on Feb 2023 test set |
| 6 | `06 visualization.ipynb` | 19 charts: scatter, residuals, feature importance, zone RMSE |

---

## 📚 References

1. Li, H., et al. (2025). *UrbanEV: An open benchmark dataset for urban electric vehicle charging demand prediction.* Scientific Data.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
3. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
4. Breiman, L. (2001). *Random Forests.* Machine Learning.
5. LangChain AI (2023). *LangGraph.* https://langchain-ai.github.io/langgraph/
6. Chroma (2023). *ChromaDB.* https://www.trychroma.com
7. Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.
8. Anthropic (2024). *Claude claude-sonnet-4-20250514.* https://www.anthropic.com

---

## 📍 Summary

**Synora combines predictive ML with agentic reasoning to enable data-driven EV infrastructure planning.**

---

## 📄 License

MIT License
