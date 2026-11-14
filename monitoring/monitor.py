"""
監視ジョブ：直近7日間の予測と実績を比べて、本番モデルが外れ始めていないかを確かめる

  1. 直近7日分の実績（販売数と気温）を読む
     本番ならPOSのデータ。このデモでは模擬データ生成スクリプトをPOSの代わりにする
  2. 同じ日の気温で、予測API（今の本番モデル）に問い合わせる
  3. 誤差（MAE）と範囲外の割合を計算する
  4. 誤差が基準を超えたら、終了コード1で終わる（K8sのJobが「失敗」になる）

使い方:
  python monitoring/monitor.py --as-of 2026-04-15 --predict-url http://localhost:8000
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data-gen"))
from generate import generate  # noqa: E402  模擬データ（POSの代わり）

DATA_START = "2025-12-01"  # 模擬データの開始日（毎回同じ日から作ると、同じ実績が再現できる）


def load_actuals(as_of: dt.date, window: int, seed: int) -> pd.DataFrame:
    """as_of を「今日」として、直近 window 日分の実績を返す"""
    df = generate(DATA_START, as_of.isoformat(), seed)
    return df.tail(window).reset_index(drop=True)


def ask_prediction(predict_url: str, date: str, max_temp: float) -> dict:
    body = json.dumps({"date": date, "max_temp": max_temp}).encode()
    req = urllib.request.Request(
        f"{predict_url}/predict", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as res:
        return json.load(res)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=os.environ.get("AS_OF") or dt.date.today().isoformat(),
                        help="「今日」とする日付。省略時は本当の今日")
    parser.add_argument("--window", type=int, default=7, help="何日分を見るか")
    parser.add_argument("--mae-threshold", type=float, default=10.0,
                        help="誤差がこれを超えたら要対応（杯）")
    parser.add_argument("--out-of-range-warn", type=float, default=0.5,
                        help="範囲外の日の割合がこれ以上なら注意報")
    parser.add_argument("--predict-url", default=os.environ.get("PREDICT_URL", "http://predict-api:8000"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    as_of = dt.date.fromisoformat(args.as_of)
    actuals = load_actuals(as_of, args.window, args.seed)

    rows = []
    for r in actuals.itertuples():
        pred = ask_prediction(args.predict_url, r.date, float(r.max_temp))
        rows.append(
            {
                "date": r.date,
                "max_temp": r.max_temp,
                "actual": int(r.sales),
                "predicted": pred["predicted_sales"],
                "error": abs(pred["predicted_sales"] - int(r.sales)),
                "out_of_range": pred["out_of_range"],
                "model_version": pred["model_version"],
            }
        )
    result = pd.DataFrame(rows)

    mae = result["error"].mean()
    oor_ratio = result["out_of_range"].mean()
    model_version = result["model_version"].iloc[0]

    print(f"=== 監視結果（今日={as_of}、直近{args.window}日、本番モデル version {model_version}） ===")
    print(result[["date", "max_temp", "actual", "predicted", "error", "out_of_range"]].to_string(index=False))
    print(f"誤差（MAE）: {mae:.1f} 杯（基準 {args.mae_threshold:.0f} 杯）")
    print(f"範囲外の割合: {oor_ratio:.0%}（注意報 {args.out_of_range_warn:.0%} 以上）")

    if oor_ratio >= args.out_of_range_warn:
        print("注意報: 学習した気温の範囲外の日が増えています。今後外れる可能性があります。")

    if mae > args.mae_threshold:
        print(f"判定: 要対応（誤差 {mae:.1f} 杯が基準 {args.mae_threshold:.0f} 杯を超えた）")
        return 1

    print("判定: 正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
