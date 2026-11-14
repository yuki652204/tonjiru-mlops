"""
豚汁の販売数を予測するモデルを学習し、MLflowに記録・登録するスクリプト

使い方:
  python training/train.py --data sales.csv --train-end 2026-03-01

  --train-end より前のデータだけを使って学習します。
  学習期間の最後の14日間は学習に使わず、「答え合わせ（検証）」に使います。
"""
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["max_temp", "is_weekend"]
TARGET = "sales"
MODEL_NAME = "tonjiru-sales"
VALIDATION_DAYS = 14


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="sales.csv")
    parser.add_argument("--train-end", required=True, help="この日付より前のデータで学習する（例: 2026-03-01）")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("tonjiru-sales")

    # 1. データを読み、学習に使う期間だけを取り出す
    df = pd.read_csv(args.data, parse_dates=["date"])
    df = df[df["date"] < args.train_end].sort_values("date")
    df[FEATURES] = df[FEATURES].astype(float)

    # 2. 最後の14日間を検証用に分ける（時系列なので、ランダムに混ぜない）
    split_date = df["date"].max() - pd.Timedelta(days=VALIDATION_DAYS - 1)
    train = df[df["date"] < split_date]
    valid = df[df["date"] >= split_date]

    with mlflow.start_run(run_name=f"train-until-{args.train_end}"):
        # 3. 学習の条件を記録する
        mlflow.log_params(
            {
                "train_start": train["date"].min().date().isoformat(),
                "train_end": args.train_end,
                "n_estimators": args.n_estimators,
                "features": ",".join(FEATURES),
            }
        )
        # 学習データの気温の範囲も残す（範囲外の日が来たら監視で気づけるように）
        mlflow.log_metrics(
            {
                "train_temp_min": float(df["max_temp"].min()),
                "train_temp_max": float(df["max_temp"].max()),
            }
        )

        # 4. 学習して、検証期間で誤差を測る
        model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=0)
        model.fit(train[FEATURES], train[TARGET])
        mae = mean_absolute_error(valid[TARGET], model.predict(valid[FEATURES]))
        mlflow.log_metric("valid_mae", mae)

        # 5. 検証期間も含めた全データで学習し直してから、モデルを登録する
        model.fit(df[FEATURES], df[TARGET])
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(df[FEATURES], model.predict(df[FEATURES])),
            registered_model_name=MODEL_NAME,
            # 自分で学習したモデルなので、決定木の内部データ型を「信頼する型」として明示する
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"学習期間: {df['date'].min().date()} 〜 {df['date'].max().date()}")
        print(f"気温の範囲: {df['max_temp'].min()}℃ 〜 {df['max_temp'].max()}℃")
        print(f"検証の誤差(MAE): {mae:.1f} 杯")
        print(f"登録: {MODEL_NAME} version {info.registered_model_version}")


if __name__ == "__main__":
    main()
