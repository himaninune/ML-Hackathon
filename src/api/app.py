from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import time
from src.config import DATA_DIR, CORS_ORIGINS
from src.api.schemas import XGBoostMetricsResponse, ViterbiMetricsResponse, MetricsOverviewResponse
from src.models.xgboost_loader import XGBoostScorer
from src.services.metrics_service import metrics_service
from src.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("pathsnap")
scorer = XGBoostScorer()
service = PredictionService(scorer)


@asynccontextmanager
async def lifespan(app):
    loaded = scorer.load()
    log.info("XGBoost load status: %s", "loaded" if loaded else scorer.error)
    yield


app = FastAPI(title="PathSnap API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=r"^\d{8}$")


def trip_ids():
    root = DATA_DIR / "student_test"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8)


@app.get("/api/health")
def health():
    return {"status": "healthy", "model_loaded": scorer.loaded,
            "model_error": None if scorer.loaded else scorer.error,
            "road_network_loaded": service.network_cache_size > 0, "backend_version": app.version}


@app.get("/api/metrics/xgboost", response_model=XGBoostMetricsResponse)
def xgboost_metrics():
    return metrics_service.xgboost()


@app.get("/api/metrics/viterbi", response_model=ViterbiMetricsResponse)
def viterbi_metrics():
    return metrics_service.viterbi()


@app.get("/api/metrics/overview", response_model=MetricsOverviewResponse)
def metrics_overview():
    return metrics_service.overview()


@app.get("/api/trips")
def list_trips():
    out = []
    for tid in trip_ids():
        try:
            gps = service.load_gps(tid)
            duration = float(gps.timestamp.max() - gps.timestamp.min()) if len(gps) else None
            out.append({"trip_id": tid, "observations": len(gps), "duration_seconds": duration})
        except Exception:
            log.exception("Unable to inspect trip %s", tid)
    return {"trips": out}


@app.get("/api/trips/{trip_id}")
def trip_details(trip_id: str):
    if trip_id not in trip_ids():
        raise HTTPException(404, "Trip not found")
    try:
        gps = service.load_gps(trip_id)
        return {"trip_id": trip_id, "observations": gps[["longitude", "latitude", "timestamp"]].to_dict("records"),
                "duration_seconds": float(gps.timestamp.max() - gps.timestamp.min()) if len(gps) else 0.0}
    except Exception as exc:
        log.exception("Trip detail failed")
        raise HTTPException(500, "Unable to load trip") from exc


@app.get("/api/trips/{trip_id}/map")
def trip_map(trip_id: str):
    if trip_id not in trip_ids():
        raise HTTPException(404, "Trip not found")
    gps = service.load_gps(trip_id)
    return {"trip_id": trip_id, "gps_coordinates": gps[["latitude", "longitude"]].values.tolist()}


@app.post("/api/predict")
def predict(request: PredictRequest):
    if request.trip_id not in trip_ids():
        raise HTTPException(404, "Trip not found")
    if not scorer.loaded:
        raise HTTPException(503, "Selected XGBoost model is unavailable")
    log.info("Prediction request received for trip %s", request.trip_id)
    started = time.perf_counter()
    try:
        result = service.predict(request.trip_id)
        result["processing_time_seconds"] = round(time.perf_counter() - started, 4)
        return result
    except Exception as exc:
        log.exception("Prediction failed for %s", request.trip_id)
        raise HTTPException(500, "Prediction could not be completed") from exc
