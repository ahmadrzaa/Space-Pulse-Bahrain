Space Pulse • Exoplanet AI

A World Away: Hunting for Exoplanets with AI — NASA Space Apps 2025

Space Pulse is a live end-to-end app for working with exoplanet survey data.

Pull real light curves from NASA/MAST (Kepler, K2, TESS)

Plot and phase-fold light curves in the browser

Run a fast BLS (Box Least Squares) baseline classifier

Train a tabular model on the NASA Exoplanet Archive KOI dataset and view metrics

This is not a mock. Every run on Analyze calls NASA/MAST via Lightkurve
. KOI training calls the NASA Exoplanet Archive TAP service live.

Stack

Frontend: Next.js 15 (App Router), TypeScript, Plotly.js

Backend: FastAPI, Uvicorn, Lightkurve, Astropy (BLS), scikit-learn, pandas, numpy

Data sources: MAST (Kepler/K2/TESS light curves), NASA Exoplanet Archive (KOI cumulative)

Repository layout
space-pulse/
├─ backend/
│  ├─ app.py               # FastAPI: /lightcurve, /classify, /phasefold, /train/koi, /metrics
│  ├─ koi_train.py         # KOI download + tabular training
│  ├─ models/              # (generated) trained model + metrics JSON
│  ├─ requirements.txt     # Python deps
│  └─ .venv/               # local virtualenv (ignored)
├─ frontend/
│  ├─ src/app/             # routes: /, /analyze, /models, /about, /team
│  ├─ src/components/      # Header, Footer, charts, search, team cards
│  ├─ src/lib/             # api.ts, types, config
│  ├─ public/              # logos & team photos
│  ├─ .env.local           # NEXT_PUBLIC_API_BASE=http://localhost:8000
│  └─ package.json
├─ .gitignore
└─ README.md

Quick start (local)
1) Backend (FastAPI)
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000


API: http://localhost:8000

2) Frontend (Next.js)
cd frontend
npm install
# Ensure:
# echo NEXT_PUBLIC_API_BASE=http://localhost:8000 > .env.local
npm run dev


App: http://localhost:3000

Demo flow (for judges)
Analyze

Pick mission + ID (examples: TESS/TIC 261136679, K2/EPIC 201505350, Kepler/KIC 11442793).

Fetch → live light curve from MAST.

Classify → BLS baseline label + details.

Phase-Fold → phase vs flux at best BLS period.

Export RAW/PHASE data as JSON or CSV.

Models

Train (KOI) → downloads KOI cumulative via TAP, trains a Gradient Boosting model.

Show metrics → ROC-AUC, average precision, confusion matrix, features used.

About / Team

Goals, limitations, roadmap, and team profiles.

API (short)

GET /lightcurve?mission={TESS|KEPLER|K2}&id={TIC|KIC|EPIC}

POST /classify → { label, probability, probs, details }

POST /phasefold → { period_days, t0_days, duration_days, bls_snr, points:[{phase,flux}] }

POST /train/koi → triggers live KOI training

GET /metrics → latest training metrics or baseline status

Known limitations

Currently a BLS baseline (fast & transparent); integrating the trained KOI model into the classify path is the next step.

K2 can require alternate authors or relaxed quality masks; we try K2SFF → EVEREST → generic, and if needed quality_bitmask="none".

No persistent cache; all fetches are live.

Performance notes

First fetch per target may take 5–20s (download + stitching).

Returned points are capped (default 8,000) for responsive charts.

Phase-fold uses the same BLS period as classify for consistency.

Reproducibility

Pin backend deps after changes:

cd backend
pip freeze > requirements.txt


Node 18+ recommended.

Frontend base URL in frontend/.env.local:

NEXT_PUBLIC_API_BASE=http://localhost:8000

Credits

NASA/MAST via Lightkurve for light curves

NASA Exoplanet Archive (TAP) for KOI cumulative

Astropy BLS implementation

Plotly.js for charts

Team

Ahmad Raza — Team Lead, AI Chatbot Engineer

Zahraa Sayed Mahmood — UI Developer

Noora Alabbasi — Frontend & User Journey

Mahdi Khalil Ebrahim, Zainab Ramadhan Ali Kadhem, Shayma Ali — Data & QA

(Photos live in frontend/public/team/ and can be updated anytime.)

License

MIT (code). Data remains under the terms of the original NASA sources.