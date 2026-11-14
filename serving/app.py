"""
豚汁の販売数を予測するAPI

どのモデルを使うかは、環境変数 MODEL_VERSION で決まります。
K8s では、この値を Git のマニフェストに書き、Argo CD で反映します。
"""
import os
import datetime as dt

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_NAME = os.environ.get("MODEL_NAME", "tonjiru-sales")
MODEL_VERSION = os.environ["MODEL_VERSION"]  # 必須。Git に書いた番号が入る
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

# 起動時に1回だけ、指定バージョンのモデルと「学習時の気温の範囲」を読み込む
client = mlflow.MlflowClient()
_version_info = client.get_model_version(MODEL_NAME, MODEL_VERSION)
_run = client.get_run(_version_info.run_id)
TRAIN_TEMP_MIN = _run.data.metrics["train_temp_min"]
TRAIN_TEMP_MAX = _run.data.metrics["train_temp_max"]
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

app = FastAPI(title="tonjiru-sales predictor")


class PredictRequest(BaseModel):
    date: dt.date = Field(description="予測したい日")
    max_temp: float = Field(description="その日の予想最高気温（℃）")


class PredictResponse(BaseModel):
    predicted_sales: int
    model_version: str
    out_of_range: bool
    warning: str | None = None


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    is_weekend = 1.0 if req.date.weekday() >= 5 else 0.0
    features = pd.DataFrame({"max_temp": [req.max_temp], "is_weekend": [is_weekend]})
    sales = round(float(model.predict(features)[0]))

    # 学習時に経験していない気温なら、仕込み前の時点で警告を出す
    out_of_range = not (TRAIN_TEMP_MIN <= req.max_temp <= TRAIN_TEMP_MAX)
    warning = None
    if out_of_range:
        warning = (
            f"予想気温 {req.max_temp}℃ は、このモデルが学習した範囲"
            f"（{TRAIN_TEMP_MIN}〜{TRAIN_TEMP_MAX}℃）の外です。予測が外れる可能性があります。"
        )

    return PredictResponse(
        predicted_sales=sales,
        model_version=MODEL_VERSION,
        out_of_range=out_of_range,
        warning=warning,
    )
