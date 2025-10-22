# Iris Backend (FastAPI)

A FastAPI service that loads the latest Iris model from Weights & Biases (W&B) and exposes endpoints for prediction and model info.

## Endpoints

- GET /health
- GET /model/info
- GET /model/validation
- POST /predict

## Run locally

- Install Python 3.11+
- Create a virtual environment and install requirements:

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

- Set W&B environment variables (API key required to fetch artifacts):

```
$env:WANDB_API_KEY = "<your-key>"
$env:WANDB_ENTITY  = "<your-entity>"   # optional if project in your default account
$env:WANDB_PROJECT = "mlops-capstone"  # or your project name
```

- Start the API:

```
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## Docker

Build and run:

```
docker build -t iris-backend .
docker run --rm -p 8000:8000 -e WANDB_API_KEY="<your-key>" -e WANDB_ENTITY="<entity>" -e WANDB_PROJECT="MLOPSPROJECT2" iris-backend
```

The service will start even if a model can't be loaded, with `status: degraded` in `/health`.
