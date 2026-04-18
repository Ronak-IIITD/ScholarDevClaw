FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pytest pytest-json-report pytest-cov \
    torch --index-url https://download.pytorch.org/whl/cpu \
    "jax[cpu]" \
    flax optax \
    numpy scipy scikit-learn pandas \
    transformers datasets tokenizers \
    fastapi uvicorn gradio \
    networkx matplotlib seaborn

WORKDIR /workspace
ENV PYTHONPATH=/workspace/src

CMD ["pytest", "tests/", "-v", "--json-report", "--json-report-file=/tmp/report.json"]
