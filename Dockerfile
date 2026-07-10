# AMD Hackathon Track 1 — batch processor (linux/amd64)
# Bundles a Qwen2.5-1.5B-Instruct Q4_K_M GGUF for in-process local inference.
FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SKIP_LOCAL=false \
    LOCAL_GGUF_PATH=/models/model.gguf \
    LOCAL_LLM_MODEL=bundled-gguf \
    REQUEST_TIMEOUT_SECONDS=30 \
    BATCH_TIMEOUT_SECONDS=600 \
    CMAKE_ARGS="-DLLAMA_BLAS=OFF"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-batch.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir llama-cpp-python

# ~1 GB Q4_K_M GGUF — fits Track 1 4 GB RAM / 10 GB image budgets.
ARG LOCAL_GGUF_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
RUN mkdir -p /models \
    && curl -L --fail --retry 3 -o /models/model.gguf "${LOCAL_GGUF_URL}" \
    && test -s /models/model.gguf

COPY my_routing_agent/ my_routing_agent/
COPY app.py run_batch.py ./

RUN mkdir -p /input /output \
    && apt-get purge -y build-essential cmake \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python", "run_batch.py"]
