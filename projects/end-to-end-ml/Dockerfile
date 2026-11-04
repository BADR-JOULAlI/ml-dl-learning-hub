FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY projects/end-to-end-ml ./projects/end-to-end-ml
RUN python -m pip install --no-cache-dir ".[api]"
ENV MODEL_PATH=/app/artifacts/churn_pipeline.joblib
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "serve:app", "--app-dir", "projects/end-to-end-ml", "--host", "0.0.0.0", "--port", "8000"]
