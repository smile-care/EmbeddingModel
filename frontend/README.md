# Data Cluster Frontend

Vue 3 + Vite + TypeScript UI for the Data Cluster platform (dataset management,
single-image mask annotation, contrastive training, and inference visualization).

It talks to the FastAPI backend in `src/data_cluster` over `/api` and `/static`,
which Vite proxies to `http://127.0.0.1:8000` during development.

## Run locally

**Prerequisites:** Node.js, and the backend running (see `../run_backend.sh`).

1. Install dependencies:
   `npm install`
2. Start the dev server:
   `npm run dev`
3. Open the printed URL (defaults to port 3000).

To point at a non-proxied backend, set `VITE_API_BASE_URL` in `.env`.
