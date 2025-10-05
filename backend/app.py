# app.py — FastAPI for live NASA/MAST light curves + BLS baseline + KOI training

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares  # transit search
from koi_train import train_koi_tabular, load_metrics  # <-- added

app = FastAPI(title="Space Pulse • Exoplanet AI API")

# allow local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- models ----------
class LCPoint(BaseModel):
    time: float
    flux: float

class LightCurveOut(BaseModel):
    mission: str
    target: str
    points: list[LCPoint]
    meta: dict | None = None

class ClassifyIn(BaseModel):
    mission: str
    id: str
    points: list[LCPoint] | None = None  # optional; we can fetch again if not provided
    options: dict | None = None

class ClassifyOut(BaseModel):
    label: str
    probability: float
    probs: dict | None = None
    details: dict | None = None

# ---------- helpers ----------
def _normalize_flux(flux: np.ndarray) -> np.ndarray:
    f = flux.astype(float)
    f = f[~np.isnan(f)]
    med = np.median(f) if f.size else 1.0
    if not np.isfinite(med) or med == 0:
        med = 1.0
    return flux / med

def _prefix_target(mission: str, ident: str) -> str:
    s = ident.strip().upper().replace("  ", " ")
    if mission == "TESS":
        return s if s.startswith("TIC") else f"TIC {s}"
    if mission == "KEPLER":
        return s if s.startswith("KIC") else f"KIC {s}"
    if mission == "K2":
        return s if s.startswith("EPIC") else f"EPIC {s}"
    return s

def _fetch_lightcurve_series(mission: str, target: str):
    """
    Robust fetch of a usable light-curve (time, flux) in numpy arrays.
    - TESS: prefer SPOC
    - Kepler: standard Kepler LC
    - K2: try K2SFF, then EVEREST, then generic; if masking kills data, retry with quality_bitmask='none'
    """
    last_exc = None

    if mission == "TESS":
        tries = [dict(mission="TESS", author="SPOC"), dict(mission="TESS")]
    elif mission == "KEPLER":
        tries = [dict(mission="Kepler")]
    else:  # K2
        tries = [
            dict(mission="K2", author="K2SFF"),
            dict(mission="K2", author="EVEREST"),
            dict(mission="K2"),
        ]

    for query in tries:
        try:
            sr = lk.search_lightcurve(target, **query)
            if sr is None or len(sr) == 0:
                continue

            # 1) normal download first
            lcs = sr.download_all()
            if lcs is None:
                continue
            lc = lcs.stitch().remove_nans()

            t = np.asarray(lc.time.value, dtype=float)
            f = np.asarray(lc.flux.value, dtype=float)

            # If too few points, retry with a looser quality mask
            if t.size < 200:
                lcs = sr.download_all(quality_bitmask="none")
                if lcs is None:
                    continue
                lc = lcs.stitch().remove_nans()
                t = np.asarray(lc.time.value, dtype=float)
                f = np.asarray(lc.flux.value, dtype=float)

            # still not enough? skip this author and try next
            if t.size < 200 or not np.isfinite(t).any() or not np.isfinite(f).any():
                continue

            f = _normalize_flux(f)
            return t, f

        except Exception as e:
            last_exc = e
            continue

    raise HTTPException(
        status_code=404,
        detail=f"No usable light curve for {target} ({mission}). "
               f"Tried multiple sources; last error: {last_exc}"
    )

# ---------- endpoints ----------
@app.get("/lightcurve", response_model=LightCurveOut)
def get_lightcurve(
    mission: str = Query(..., pattern="^(TESS|KEPLER|K2)$"),
    id: str = Query(..., min_length=1),
    max_points: int = Query(8000, ge=100, le=20000),
):
    target = _prefix_target(mission, id)
    try:
        t, f = _fetch_lightcurve_series(mission, target)
        if t.size > max_points:
            idx = np.linspace(0, t.size - 1, num=max_points, dtype=int)
            t = t[idx]; f = f[idx]
        points = [LCPoint(time=float(ti), flux=float(fi)) for ti, fi in zip(t, f)]
        meta = {"n_points": len(points), "mission": mission}
        return {"mission": mission, "target": target, "points": points, "meta": meta}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch failed: {e}")
    


@app.post("/classify", response_model=ClassifyOut)
def classify(req: ClassifyIn):
    """
    Simple, fast baseline: Box Least Squares (BLS) transit search.
    Computes best period & SNR over a small duration grid, then maps SNR -> probability (logistic).
    """
    mission = req.mission.upper()
    target = _prefix_target(mission, req.id)

    # get series
    try:
        if req.points and len(req.points) > 100:
            t = np.array([p.time for p in req.points], dtype=float)
            f = np.array([p.flux for p in req.points], dtype=float)
        else:
            t, f = _fetch_lightcurve_series(mission, target)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch failed: {e}")

    # basic cleanup
    mask = np.isfinite(t) & np.isfinite(f)
    t, f = t[mask], f[mask]

    # detrend (very light)
    if f.size > 500:
        k = 201
        med = np.convolve(f, np.ones(k)/k, mode="same")
        med[~np.isfinite(med)] = 1.0
        med[med == 0] = 1.0
        f = f / med

    # -------- BLS search (multi-duration for robustness) --------
    min_period = 0.5
    max_period = 20.0
    n_periods = 2000
    periods = np.linspace(min_period, max_period, n_periods)

    # try a few plausible transit durations (days)
    duration_grid = np.array([0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30])

    bls = BoxLeastSquares(t, f)
    best = {"snr": -np.inf}
    for dur in duration_grid:
        res = bls.power(periods, dur, objective="snr")
        i = int(np.nanargmax(res.power))
        snr = float(res.power[i])
        if snr > best["snr"]:
            best.update({
                "i": i,
                "period": float(res.period[i]),
                "snr": snr,
                "depth": float(res.depth[i]) if hasattr(res, "depth") else float("nan"),
                "duration": float(dur),
                "t0": float(res.transit_time[i]) if hasattr(res, "transit_time") else float(t[0]),
            })

    best_period = best["period"]
    best_snr = best["snr"]
    depth = best["depth"]
    best_duration = best["duration"]
    # t0 is not used here further; kept for parity with /phasefold

    # -------- map SNR to pseudo-probability (softer sigmoid) --------
    def sigmoid(x): 
        return 1 / (1 + np.exp(-(x - 5.0)/2.0))

    prob_candidate = float(np.clip(sigmoid(best_snr), 0, 1))
    prob_fp = float(1 - prob_candidate)
    label = "candidate" if prob_candidate >= 0.5 else "false_positive"

    return {
        "label": label,
        "probability": prob_candidate if label == "candidate" else prob_fp,
        "probs": {"candidate": prob_candidate, "false_positive": prob_fp},
        "details": {
          "best_period_days": best_period,
          "bls_snr": best_snr,
          "depth": depth,
          "duration_days": best_duration,
          "n_points": int(t.size),
          "note": "BLS baseline — final ML model to be added."
        }
    }

# ----- NEW: Live training on NASA KOI data -----
@app.post("/train/koi")
def train_koi():
    """
    Live training:
    - Downloads the Kepler KOI cumulative CSV from NASA Exoplanet Archive
    - Trains a small tabular model
    - Saves metrics/model to backend/models/
    """
    res = train_koi_tabular()
    return {"status": "ok", "metrics": res.metrics}

@app.get("/metrics")
def metrics():
    m = load_metrics()
    return m or {
        "model": None,
        "status": "baseline_only",
        "note": "Run POST /train/koi (from the Models page) to download KOI labels and train.",
    }

# ---------- Phase-folded view (BLS best period) ----------
from pydantic import BaseModel as _BaseModel  # reuse pydantic

class _PhasePoint(_BaseModel):
    phase: float  # normalized to [-0.5, 0.5]
    flux: float

class _PhaseFoldOut(_BaseModel):
    mission: str
    target: str
    period_days: float
    t0_days: float
    duration_days: float
    bls_snr: float
    points: list[_PhasePoint]

@app.post("/phasefold", response_model=_PhaseFoldOut)
def phasefold(req: ClassifyIn):
    """
    Compute best BLS period using a small duration grid, then return a phase-folded series for plotting.
    Phase normalized to [-0.5, 0.5] for centered transit visualization.
    """
    mission = req.mission.upper()
    target = _prefix_target(mission, req.id)

    # get series
    try:
        if req.points and len(req.points) > 100:
            t = np.array([p.time for p in req.points], dtype=float)
            f = np.array([p.flux for p in req.points], dtype=float)
        else:
            t, f = _fetch_lightcurve_series(mission, target)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch failed: {e}")

    mask = np.isfinite(t) & np.isfinite(f)
    t, f = t[mask], f[mask]

    # light detrend as in /classify
    if f.size > 500:
        k = 201
        med = np.convolve(f, np.ones(k)/k, mode="same")
        med[~np.isfinite(med)] = 1.0
        med[med == 0] = 1.0
        f = f / med

    # -------- BLS search (multi-duration, same as /classify) --------
    min_period = 0.5
    max_period = 20.0
    n_periods = 2000
    periods = np.linspace(min_period, max_period, n_periods)

    duration_grid = np.array([0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30])

    bls = BoxLeastSquares(t, f)
    best = {"snr": -np.inf}
    for dur in duration_grid:
        res = bls.power(periods, dur, objective="snr")
        i = int(np.nanargmax(res.power))
        snr = float(res.power[i])
        if snr > best["snr"]:
            best.update({
                "i": i,
                "period": float(res.period[i]),
                "snr": snr,
                "depth": float(res.depth[i]) if hasattr(res, "depth") else float("nan"),
                "duration": float(dur),
                "t0": float(res.transit_time[i]) if hasattr(res, "transit_time") else float(t[0]),
            })

    period = best["period"]
    t0 = best["t0"]
    best_snr = best["snr"]
    dur = best["duration"]

    # phase in [-0.5, 0.5]
    raw_phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    phase = raw_phase / period

    # optional downsample for transport/plotting
    max_points = 8000
    if phase.size > max_points:
        idx = np.linspace(0, phase.size - 1, num=max_points, dtype=int)
        phase = phase[idx]; f = f[idx]

    pts = [_PhasePoint(phase=float(ph), flux=float(fl)) for ph, fl in zip(phase, f)]

    return {
        "mission": mission,
        "target": target,
        "period_days": period,
        "t0_days": t0,
        "duration_days": dur,
        "bls_snr": best_snr,
        "points": pts,
    }

# ---------- health ----------
@app.get("/health")
def health():
    return {"status": "ok"}
