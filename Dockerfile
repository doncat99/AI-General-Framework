# ---------- Base image ----------
FROM python:3.11.14-slim-bookworm

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# If you truly need no system packages, keep this empty.
# Uncomment if you later need OS libs: gcc, libpq, etc.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# ----- Hardcoded Python deps (ONLY what you want) -----
# Web server runtime
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install alembic==1.17.0 asyncpg==0.30.0 bcrypt==4.3.0 \
     "fastapi[all]"==0.116.2 pyjwt==2.10.1 \
     "sqlalchemy[asyncio]"==2.0.44 uvloop==0.22.1 neo4j==5.28.2

ENV PATH="/venv/bin:$PATH"

# ----- App files -----
COPY app app
COPY alembic alembic
COPY alembic.ini .
COPY init.sh .
RUN chmod +x ./init.sh

# ---------- Ports ----------
EXPOSE 0

# ---------- Entrypoint & CMD ----------
ENTRYPOINT ["./init.sh"]
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8002", "--workers", "2", "--loop", "uvloop"]
