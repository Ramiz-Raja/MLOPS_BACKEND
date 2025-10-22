# Backend Dockerfile (FastAPI)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# App code
COPY . /app

EXPOSE 8000

# W&B environment (set at runtime or via .env if mounted)
# ENV WANDB_PROJECT="MLOPSPROJECT2"
# ENV WANDB_ENTITY="your-wandb-entity"
# ENV WANDB_API_KEY="your-wandb-api-key"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
