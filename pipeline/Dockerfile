FROM python:3.13-slim

WORKDIR /app
# 監視に加えて、再学習（scikit-learn）と MLflow への登録も行うので、学習と同じライブラリを入れる
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data-gen/ data-gen/
COPY training/ training/
COPY monitoring/monitor.py monitoring/monitor.py
COPY pipeline/ pipeline/

RUN useradd --create-home appuser
USER appuser

CMD ["sh", "pipeline/run.sh"]
