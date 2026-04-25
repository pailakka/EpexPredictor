FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code/

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY ./predictor /code/predictor

ENV PATH="/code/.venv/bin:$PATH"
CMD ["uvicorn", "predictor.api.priceapi:app", "--host", "0.0.0.0", "--port", "80"]
