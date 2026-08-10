# Local Prometheus + Grafana monitoring

## What is scraped

| Target | URL | Notes |
|---|---|---|
| API | `http://api:8000/metrics` | FastAPI + instrumentator + custom Kepler metrics |
| Worker | `http://worker:9100/metrics` | Training CPU/RSS and job counters |

## Key metrics

- `kepler_http_5xx_total` — HTTP 500+ responses (also `http_requests_total{status=~"5.."}`)
- `kepler_inference_duration_seconds` / `kepler_inference_requests_total`
- `kepler_training_duration_seconds` / `kepler_training_jobs_total`
- `kepler_model_op_peak_rss_bytes` / `kepler_model_op_cpu_percent` — peaks during inference/training

## Bring up

```powershell
docker compose up --build -d
```

If host ports collide with existing containers (common: Redis `6379`, registry/`mlflow` `5000`, local API `8000`), set overrides in `.env` — for example:

```env
KEPLER_API_HOST_PORT=8001
KEPLER_MLFLOW_HOST_PORT=5001
KEPLER_REDIS_HOST_PORT=6380
KEPLER_PROMETHEUS_HOST_PORT=9090
KEPLER_GRAFANA_HOST_PORT=3000
```

- Grafana: http://localhost:3000 (`admin` / `admin`) — dashboard **Kepler Engine**
- Prometheus: http://localhost:9090
- API metrics: http://localhost:${KEPLER_API_HOST_PORT:-8000}/metrics
- Worker metrics: http://localhost:${KEPLER_WORKER_METRICS_HOST_PORT:-9100}/metrics

Host ports are overridable via `KEPLER_GRAFANA_HOST_PORT`, `KEPLER_PROMETHEUS_HOST_PORT`, `KEPLER_WORKER_METRICS_HOST_PORT` in `.env`.
