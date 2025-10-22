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
$env:WANDB_PROJECT = "MLOPSPROJECT2"   # just the project name (no slashes)
# Optional: provide a full artifact reference to override entity/project/model/alias
# Format: [entity/]<project>/<artifact_name>:<alias>
# $env:WANDB_ARTIFACT = "raja-ramiz-mukhtar6-szabist/MLOPSPROJECT2/iris-logreg-model:latest"
```

- Start the API:

```
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Alternatively, create a `.env` file in the project root with:

```
WANDB_API_KEY=<your-key>
WANDB_ENTITY=raja-ramiz-mukhtar6-szabist
WANDB_PROJECT=MLOPSPROJECT2
# Optional explicit artifact override
# WANDB_ARTIFACT=raja-ramiz-mukhtar6-szabist/MLOPSPROJECT2/iris-logreg-model:latest
```

## Docker

Build and run:

```
docker build -t iris-backend .
docker run --rm -p 8000:8000 -e WANDB_API_KEY="<your-key>" -e WANDB_ENTITY="<entity>" -e WANDB_PROJECT="MLOPSPROJECT2" iris-backend
```

The service will start even if a model can't be loaded, with `status: degraded` in `/health`.

### Notes on W&B configuration

- Ensure `WANDB_PROJECT` contains only the project name (e.g., `MLOPSPROJECT2`). Do not include the entity or repeat the project (avoid strings like `entity/project` or `project/project`). The backend now auto-normalizes common mistakes, but correct values are recommended.
- If your project or artifact naming differs, set `WANDB_ARTIFACT` to the exact artifact path, for example:

```
$env:WANDB_ARTIFACT = "<entity>/<project>/iris-logreg-model:production"
```

This will be used directly instead of composing from entity/project.
