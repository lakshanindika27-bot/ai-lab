ARG PY=3.12
FROM python:${PY}-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY 02-structured 02-structured
COPY 05-cost-cache 05-cost-cache
COPY 06-serve 06-serve
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "--app-dir", "06-serve", "server:app", "--host", "0.0.0.0", "--port", "8000"]
