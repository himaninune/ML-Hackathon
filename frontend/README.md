# PathSnap frontend

React, TypeScript, Vite, Tailwind CSS, Leaflet, and Recharts client for the PathSnap FastAPI service.

## Run locally

From this directory, install dependencies and start Vite:

```powershell
npm install
Copy-Item .env.example .env.local
npm run dev -- --host 0.0.0.0 --port 3000
```

Set `VITE_API_BASE_URL` in `.env.local` to the backend base URL, such as `http://localhost:8000`. The frontend reads the actual trip and prediction responses from the API; its prediction activity history is kept only in the current browser session.

## Two-laptop setup

On Laptop 1, run the FastAPI service on `0.0.0.0:8000` and allow inbound TCP port 8000 through its firewall. Find Laptop 1's local IPv4 address with `ipconfig`. On Laptop 2, set `VITE_API_BASE_URL=http://<LAPTOP_1_IPV4>:8000`, then run the frontend on port 3000 and open `http://localhost:3000` in Laptop 2's browser. Allow Laptop 2 to reach Laptop 1 on the local network.

FastAPI CORS must allow the exact browser origin. For this setup that is `http://localhost:3000`; if the browser opens Vite through another host or port, configure that origin in the backend's `PATHSNAP_CORS_ORIGINS` setting. Restart Vite after changing `.env.local`.

## Checks

```powershell
npm run typecheck
npm run build
```

The trip selector and dashboard reflect the trips returned by the backend. The backend currently returns student-testing trips; the UI does not provide reference routes. Evaluation is labelled heuristic when the API identifies it as heuristic, and unavailable when the API has no evaluation data.

The Evaluation page fetches `GET /api/metrics/overview` and keeps the XGBoost candidate-classification results separate from HMM/Viterbi route-reconstruction results. The Viterbi artifact is from student-test data with heuristic route-membership labels; the page labels it as not official ground truth. Missing or invalid metric files are shown as unavailable by the backend and are not replaced with frontend values.
