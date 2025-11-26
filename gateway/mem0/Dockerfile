# Mem0 service: Python slim
FROM python:3.11.14-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
WORKDIR /app

# (Optional) native deps if you need to build wheels
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential curl git \
#  && rm -rf /var/lib/apt/lists/*

# Mem0 + extras
RUN pip install "mem0ai[graph]"==0.1.118
# Add Ollama client for local embeddings provider support
RUN pip install ollama==0.6.0

# Install additional dependencies for vector databases and API
RUN pip install loguru==0.7.3 pydantic==2.12.3 chromadb==1.2.0 "fastapi[all]"==0.116.2 uvloop==0.22.1

# Recreate package path `gateway.mem0` in the image and copy THIS folder
# (compose build context should be ./gateway/mem0)
RUN mkdir -p /app/gateway/mem0
COPY . /app/gateway/mem0

# Make sure the package imports cleanly
RUN touch /app/gateway/__init__.py /app/gateway/mem0/__init__.py

# Persistent dirs for Mem0/Chroma
RUN mkdir -p /app/config /app/data
ENV MEM0_DATA_DIR=/app/data \
    MEM0_CONFIG_DIR=/app/config

EXPOSE 8001
# Bind uvicorn to 8001 inside the container
CMD ["uvicorn", "gateway.mem0.service:app", "--host", "0.0.0.0", "--port", "8001"]
