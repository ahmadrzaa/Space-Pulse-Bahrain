<table>
  <tr>
    <td>

<h1>🚀 Space Pulse • Exoplanet AI <br>(Team-SpacePluseBahrain)</h1>

<b>Challenge: A World Away: Hunting for Exoplanets with AI — NASA Space Apps 2025</b>  
Space Pulse is a live end-to-end web app that uses real NASA data to analyze exoplanets using AI and interactive visual tools.

---

<h2>🌌 Features</h2>

- Pull real light curves from NASA/MAST (Kepler, K2, TESS)  
- Plot and phase-fold light curves directly in the browser  
- Run a fast BLS (Box Least Squares) baseline classifier  
- Train a tabular ML model on the NASA KOI dataset  
- Live, end-to-end API integrations — nothing is mocked

---

<h2>🧰 Technologies Used</h2>

<b>Frontend:</b> Next.js 15 (App Router), TypeScript, Plotly.js  
<b>Backend:</b> FastAPI, Uvicorn, Lightkurve, Astropy (BLS), scikit-learn, pandas, numpy  
<b>Data:</b> NASA/MAST (Kepler/K2/TESS), NASA Exoplanet Archive (KOI cumulative)

---

<h2>📁 Project Structure</h2>

<pre>
space-pulse/
├─ backend/
│  ├─ app.py — FastAPI: /lightcurve, /classify, /phasefold, /train/koi, /metrics
│  ├─ koi_train.py — KOI download + model training
│  ├─ models/ — trained model + metrics
│  └─ requirements.txt
├─ frontend/
│  ├─ src/app/ — routes: /, /analyze, /models, /about, /team
│  ├─ components/, lib/, public/
│  ├─ .env.local → NEXT_PUBLIC_API_BASE=http://localhost:8000
│  └─ package.json
</pre>

---

<h2>⚡ Quick Start (Local)</h2>

<b>Backend:</b>  
<pre>
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
</pre>

<b>Frontend:</b>  
<pre>
cd frontend
npm install
echo NEXT_PUBLIC_API_BASE=http://localhost:8000 > .env.local
npm run dev
</pre>

---

<h2>🧪 Demo Flow (Judges Walkthrough)</h2>

<b>Analyze:</b>  
Choose mission + ID (e.g. TESS/TIC 261136679)  
→ Fetch light curve  
→ Classify with BLS  
→ Phase-Fold  
→ Export results (CSV/JSON)

<b>Models:</b>  
Train on KOI data → Live model training  
→ View metrics: ROC-AUC, precision, confusion matrix

<b>About / Team:</b>  
Goals, limitations, roadmap, bios

---

<h2>📡 API Overview</h2>

- GET `/lightcurve?mission=TESS&ID=TIC`  
- POST `/classify` → `{ label, probs, details }`  
- POST `/phasefold` → `{ period_days, flux, phase, snr }`  
- POST `/train/koi` → triggers training  
- GET `/metrics` → returns model metrics

---

<h2>⚠ Known Limitations</h2>

- BLS baseline model only (KOI model integration coming)  
- No persistent cache (all fetches live)  
- K2 requires fallback handling  
- First-time fetch takes ~5–20s  
- Chart points capped to ~8,000 for speed

---

<h2>♻ Reproducibility</h2>

- Use `pip freeze > requirements.txt` after backend updates  
- Node 18+ recommended  
- Set API base in: `frontend/.env.local` →  
  `NEXT_PUBLIC_API_BASE=http://localhost:8000`

---

<h2>🙌 Credits</h2>

- NASA/MAST via Lightkurve  
- NASA Exoplanet Archive (TAP)  
- Astropy BLS  
- Plotly.js

---

<h2>👥 Team</h2>

- <b>Ahmad Raza</b> — Team Lead, AI Chatbot Engineer  
- <b>Zahraa Sayed Mahmood</b> — UI Developer  
- <b>Nasser Zainalabedin</b> — Frontend & User Journey  
- <b>Amina Kashfi</b>, <b>Deena Al Malki</b>, <b>Mohammed Hejairi</b> — Data & QA



---

<h2>📄 License</h2>

MIT License (code).  
Data usage follows NASA source terms.

   
  </tr>
</table>
