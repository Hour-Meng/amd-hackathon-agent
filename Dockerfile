# AMD Hackathon Track 1 — batch processor (linux/amd64)
FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SKIP_LOCAL=true \
    REQUEST_TIMEOUT_SECONDS=30 \
    BATCH_TIMEOUT_SECONDS=600

COPY requirements-batch.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY my_routing_agent/ my_routing_agent/
COPY app.py run_batch.py ./

RUN mkdir -p /input /output

ENTRYPOINT ["python", "run_batch.py"]
