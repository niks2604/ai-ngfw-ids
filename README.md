# AI-NGFW: Intelligent Next-Generation Firewall with ML-based IDS

> Ensemble machine-learning firewall with a three-tier decision engine, Zero
> Trust policy layer, SHAP-based explainability, and a real-time React
> control plane.

AI-NGFW inspects network flows in-line and decides, per flow, whether to
**ALLOW**, **INSPECT**, or **BLOCK** based on an ensemble of three
complementary models (Random Forest, XGBoost, IsolationForest). Every
decision is auditable: the API returns the top SHAP contributors and a
plain-English rationale ("BLOCKED because: Forward Bytes (-0.08), Flow
IAT Max (+0.05), …"), plus a Zero Trust recommendation block that maps
the model score onto concrete actions (step-up auth, rate limit,
honeypot redirect, quarantine).

---

## Novel contributions

1. **Sub-millisecond inference path.** The ensemble runs in-process behind a
   single FastAPI service — no network hop between scoring and enforcement.
   Typical `/predict` round-trip is well under the threshold needed for
   per-flow enforcement on commodity hardware.

2. **Three-tier decision engine (ALLOW / INSPECT / BLOCK) with tunable
   thresholds.** Two cut-points (`0.30` and `0.70` by default) replace the
   binary allow/deny of classical firewalls, giving SOC operators a middle
   band for DPI / sandbox routing without dropping traffic.

3. **Honeypot feedback loop.** Flows crossing the BLOCK threshold or flagged
   CRITICAL by the Zero Trust layer are captured by `HoneypotManager` into a
   crash-safe JSONL event stream with a pluggable backend interface
   (Cowrie placeholder included) — turning blocked attackers into labelled
   training data.

4. **Explainable AI at inference time, not after the fact.** `/predict/explain`
   returns signed, weighted TreeSHAP contributions from the RF + XGBoost
   members of the ensemble plus a human-readable rationale. The Zero Trust
   layer surfaces the principles it applied (*Never Trust Always Verify*,
   *Assume Breach*, *Continuous Verification*, *Least Privilege*) alongside
   each decision.

---

## Tech stack

**Backend**
- Python 3.13
- FastAPI + Uvicorn (inference service)
- scikit-learn (Random Forest, IsolationForest)
- XGBoost (gradient-boosted classifier)
- SHAP (TreeExplainer for RF + XGBoost)
- pandas / NumPy / PyArrow (feature pipeline + parquet I/O)
- joblib (model serialisation)

**Frontend**
- React 18 + Vite
- Tailwind CSS (dark glassmorphism theme)
- Recharts (area / pie / bar / heatmap)
- Lucide React (icons)
- React Router

**Data**
- CICIDS2017 (training + evaluation + demo replay)
- NSL-KDD (cross-dataset validation)

---

## Installation

### Prerequisites
- Python 3.11+ (3.13 recommended)
- Node.js 20+ with npm 10+
- ~4 GB free disk for trained models + processed dataset

### Backend

```bash
git clone https://github.com/niks2604/ai-ngfw-ids.git
cd ai-ngfw-ids

python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Trained models and the processed CICIDS2017 dataset are **not** committed
(too large for git). You'll need either:

- Your own `trained_models/` from running the pipeline (see below), or
- Pre-trained artifacts dropped into `trained_models/` with filenames:
  `random_forest.joblib`, `xgboost.joblib`, `isolation_forest.joblib`,
  `scaler.joblib`, `feature_columns.joblib`, `label_encoder.joblib`.

### Dataset

```bash
# NSL-KDD — pulled directly from the defcom17 mirror (no auth required)
mkdir -p data/nslkdd && cd data/nslkdd
for f in "KDDTrain+.txt" "KDDTest+.txt" "KDDTrain+_20Percent.txt" "KDDTest-21.txt" "Field%20Names.csv"; do
  curl -sSfL -o "$(echo $f | sed 's/%20/ /g')" \
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/$f"
done
cd ../..

# CICIDS2017 — download from https://www.unb.ca/cic/datasets/ids-2017.html
# and place the parquet files under data/cicids2017/
```

### Frontend

```bash
cd app/frontend
npm install
```

---

## Usage

### Run the full stack

**Terminal 1 — API backend:**
```bash
cd /path/to/ai-ngfw-ids
source venv/bin/activate
uvicorn app.api.main:app --reload
# -> http://localhost:8000, interactive docs at /docs
```

**Terminal 2 — React dashboard:**
```bash
cd /path/to/ai-ngfw-ids/app/frontend
npm run dev
# -> http://localhost:5173
```

Open the dashboard, navigate to the **Overview** page, find the **Demo
Simulator** card, pick a scenario (e.g. *DDoS* or *Mixed*), set speed
(flows / second, 1×–100×), and click **Start**. The stat cards, threat
timeline, Live Flows table, SHAP explainability chart, and threat
analytics heatmap all populate in real time.

### API endpoints

| Method | Path                 | Purpose                                               |
|--------|----------------------|-------------------------------------------------------|
| GET    | `/health`            | Service + model status                                |
| POST   | `/predict`           | Single flow → decision + score + ZT recommendations   |
| POST   | `/predict/batch`     | Up to 10,000 flows per request                        |
| POST   | `/predict/explain`   | Prediction + top-5 SHAP contributions + rationale     |
| POST   | `/demo/start`        | Start traffic simulator `{scenario, speed}`           |
| POST   | `/demo/stop`         | Stop traffic simulator                                |
| GET    | `/demo/status`       | Simulator status                                      |
| GET    | `/demo/recent`       | Recent scored flows (ring buffer)                     |
| GET    | `/demo/stats`        | Aggregated decisions / timeline / heatmap             |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {"Flow Duration": 4000000, "Total Fwd Packets": 50, "...": 0},
    "context":  {"src_ip": "203.0.113.5", "asset_sensitivity": "high"}
  }'
```

---

## Project structure

```
ai-ngfw-ids/
├── app/
│   ├── api/
│   │   └── main.py                   # FastAPI inference service
│   ├── models/
│   │   ├── ensemble.py               # Weighted ensemble (RF + XGB + IF)
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── isolation_forest.py
│   │   └── autoencoder.py
│   ├── features/
│   │   └── feature_extractor.py      # Feature selection + scaling
│   ├── explainability/
│   │   └── shap_explainer.py         # TreeSHAP over RF + XGB
│   ├── zero_trust/
│   │   └── zero_trust.py             # Policy engine (5 trust bands, 8 actions)
│   ├── honeypot/
│   │   └── honeypot_manager.py       # JSONL capture + Cowrie placeholder
│   ├── ingestion/
│   │   └── netflow_collector.py      # NetFlow v5 parser + UDP listener
│   ├── simulator/
│   │   └── traffic_simulator.py      # CICIDS2017 replay for demos
│   └── frontend/                     # React 18 + Vite + Tailwind + Recharts
│       ├── src/pages/                # Overview, LiveFlows, Explainability, ThreatAnalytics
│       ├── src/components/           # Sidebar, Topbar, StatCard, DecisionBadge, DemoControls
│       └── src/lib/api.js            # Fetch wrappers (proxied through Vite)
├── training/
│   ├── data_preprocessing.py
│   ├── train_all_models.py
│   └── retrain_proper.py
├── data/                             # (git-ignored) datasets live here
├── trained_models/                   # (git-ignored) serialised models
├── logs/                             # (git-ignored) honeypot + runtime logs
├── requirements.txt
└── README.md
```

---

## Datasets

### CICIDS2017
Canadian Institute for Cybersecurity's Intrusion Detection Evaluation
Dataset — labelled network flows covering Benign traffic plus seven attack
families (DDoS, DoS, Bruteforce, Portscan, WebAttacks, Infiltration,
Botnet). Used for training and demo replay.

- Source: https://www.unb.ca/cic/datasets/ids-2017.html
- Format: parquet, ~270 MB
- Files: one per attack day (`DDoS-Friday-*.parquet`, `Benign-Monday-*.parquet`, …)

### NSL-KDD
Refined version of the KDD'99 dataset — used for cross-dataset
validation (training on CICIDS2017, evaluating on NSL-KDD).

- Mirror: https://github.com/defcom17/NSL_KDD
- Files: `KDDTrain+.txt`, `KDDTest+.txt`, `KDDTrain+_20Percent.txt`,
  `KDDTest-21.txt`, `Field Names.csv`

---

## Team

_To be filled in._

---

## License

_To be decided._
