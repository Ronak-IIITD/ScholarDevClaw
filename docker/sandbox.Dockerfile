FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pytest \
    pytest-json-report \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy \
    "jax[cpu]" \
    transformers \
    datasets \
    scikit-learn \
    fastapi \
    uvicorn \
    gradio

WORKDIR /workspace

CMD ["pytest", "tests/", "-v"]
