# PathSnap backend

## What is implemented

FastAPI lives under `src/api`, with request handling separate from prediction in
`src/services/prediction_service.py`. Existing preprocessing (`src/preprocessing`),
feature contract (`src/features/build_features.py`), candidate generation
(`src/candidate_generation/generate_candidates.py`), and the discovered XGBoost
JSON artifact are reused. The feature order is the existing 12-column
`MODEL_FEATURES` list. It has no scaler or angular sin/cos transform; inference
preserves the training values and relies on XGBoost's native missing-value rules.

Each exposed trip has its own directed network, so the service lazily loads and
caches each network. Candidate emissions combine the selected model probability
with a Gaussian penalty for GPS-to-road distance (25 m scale). Log-space Viterbi
uses shortest directed road paths, GPS displacement, elapsed time, and a travel
speed penalty for transitions. Missing candidate sets remain unmatched; gaps over
the preprocessing 5-minute segment threshold start a fresh decode segment. This
is a route reconstruction stage, not independent candidate classification.

The repository contains 80 training trips and 10 `student_test` trips, but no
separate evaluator partition. Only `student_test` trips are listed by the API.
The student `.route` files contain ordered route arcs and can be used for a
server-side, post-prediction comparison. They are never read by candidate
generation or model inference. Existing training labels are explicitly noted as
route-membership heuristics, not verified GPS-point ground truth; evaluation
therefore identifies its reference as `route_membership_heuristic`. No evaluator
trip or separate evaluator metrics are claimed here.

## Endpoints

- `GET /api/health`: runtime XGBoost and cached-network status.
- `GET /api/trips`: IDs, GPS observation counts, and duration for student-test trips.
- `GET /api/trips/{trip_id}`: GPS coordinates and timestamps only.
- `GET /api/trips/{trip_id}/map`: compact GPS geometry for map display.
- `POST /api/predict` with `{"trip_id":"00000000"}`: candidate scoring, Viterbi
  road route (including intermediate connected arcs), per-observation selected
  arc geometries, GPS geometry, unmatched observations, and
  server-side route comparison when the student reference is present.

Swagger UI: `http://<laptop-1-ip>:8000/docs`; OpenAPI JSON: `/openapi.json`.
The API never returns a reference route. The optional map endpoint currently
returns GPS only; predicted road geometry is included in the prediction response.

## Configuration

Copy `.env.example` to `.env` as a reference and set these variables in the shell
or deployment environment (the application does not auto-read `.env`):

- `XGBOOST_MODEL_PATH`: selected XGBoost JSON artifact.
- `PATHSNAP_DATA_DIR`: root containing `student_test` and `train`.
- `PATHSNAP_PROCESSED_DIR`: processed GPS CSV root.
- `PATHSNAP_HOST`: defaults to `0.0.0.0` for LAN access.
- `PATHSNAP_PORT`: defaults to `8000`.
- `PATHSNAP_CORS_ORIGINS`: comma-separated frontend origins.

## Windows: run on Laptop 1

Run PowerShell from the `ML-Hackathon` project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-backend.txt
$env:XGBOOST_MODEL_PATH = (Resolve-Path 'experiments\model_results\xgboost\xgboost_baseline.json').Path
$env:PATHSNAP_DATA_DIR = (Resolve-Path 'data\Student folder').Path
$env:PATHSNAP_PROCESSED_DIR = (Resolve-Path 'data\processed').Path
$env:PATHSNAP_HOST = '0.0.0.0'
$env:PATHSNAP_PORT = '8000'
$env:PATHSNAP_CORS_ORIGINS = 'http://localhost:3000,http://127.0.0.1:3000'
uvicorn src.api.main:app --host $env:PATHSNAP_HOST --port $env:PATHSNAP_PORT
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
once, then activate again. To find Laptop 1's LAN IPv4 address, run
`ipconfig` and use the IPv4 address for the active Wi-Fi/Ethernet adapter. Allow
the listener through Windows Firewall from an elevated PowerShell prompt:

```powershell
New-NetFirewallRule -DisplayName 'PathSnap API 8000' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private
```

Keep both laptops on the same trusted network. From Laptop 2, open
`http://<laptop-1-ip>:8000/docs`, or call
`POST http://<laptop-1-ip>:8000/api/predict` with JSON body
`{"trip_id":"00000000"}`. Add Laptop 2's exact frontend origin to
`PATHSNAP_CORS_ORIGINS` and restart the backend if the browser blocks requests.

## Tests and limitations

Install the backend requirements, then run `pytest`. Unit tests cover Viterbi,
feature validation/model artifact handling, and API contracts. API integration
tests use a student-test trip only; they skip the full prediction check if the
selected artifact cannot load. The artifact currently exists at the configured
default path; startup reports a clear unavailable state rather than substituting
another model. Candidate generation remains CPU-based and trip networks are
cached lazily. The reference routes do not constitute an independent evaluator
set and their semantics should be confirmed before publishing route scores.
