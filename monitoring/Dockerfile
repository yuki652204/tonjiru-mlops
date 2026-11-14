FROM python:3.13-slim

WORKDIR /app
# 監視ジョブは予測APIに問い合わせるだけなので、MLflow や scikit-learn は入れない（イメージを小さく保つ）
RUN pip install --no-cache-dir pandas numpy

COPY data-gen/generate.py data-gen/generate.py
COPY monitoring/monitor.py monitoring/monitor.py

RUN useradd --create-home appuser
USER appuser

CMD ["python", "monitoring/monitor.py"]
