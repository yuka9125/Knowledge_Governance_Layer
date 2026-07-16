FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV APPROVED_KNOWLEDGE_PATH=data/approved_knowledge.json

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/serving
RUN mkdir -p /app/data \
    && cp /app/serving/data/approved_knowledge.json /app/data/approved_knowledge.json

EXPOSE 8000

CMD ["sh", "-c", "uvicorn serving.web_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
