"""
豚汁の販売数（模擬データ）を生成するスクリプト

業務知識にもとづくルールで作った模擬データです。
ねらいは予測精度ではなく、ドリフト → 再学習 → GitOps再デプロイの仕組みを示すことです。

ルール:
  - 冬（最高気温5℃）の平日は約100杯売れる
  - 15℃を超えたあたりから急に売れなくなり、20℃を超えると常連の約15杯だけになる
  - 土日は平日の1.3倍
  - 日ごとに±10%程度のばらつき
"""
import argparse
import math

import numpy as np
import pandas as pd

BASE_MIN = 15      # 暑い日でも買ってくれる常連の杯数
BASE_EXTRA = 85    # 寒い日に上乗せされる杯数
TURN_TEMP = 15.0   # 売れ行きが急に落ちる境目の気温（℃）
STEEPNESS = 2.5    # 落ち方の急さ（小さいほど急）
WEEKEND_RATE = 1.3


def expected_sales(temp: float, is_weekend: bool) -> float:
    """気温と曜日から、ばらつき抜きの販売数を計算する"""
    sales = BASE_MIN + BASE_EXTRA / (1 + math.exp((temp - TURN_TEMP) / STEEPNESS))
    if is_weekend:
        sales *= WEEKEND_RATE
    return sales


def daily_max_temp(day_of_year: int, rng: np.random.Generator) -> float:
    """1年を通した最高気温（名古屋の気候を大まかにまねた模擬値）"""
    # 1月下旬が最も寒く（約9℃）、8月上旬が最も暑い（約33℃）
    seasonal = 21 - 12 * math.cos(2 * math.pi * (day_of_year - 25) / 365)
    return round(seasonal + rng.normal(0, 2.5), 1)


def generate(start: str, end: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for date in pd.date_range(start, end, freq="D"):
        temp = daily_max_temp(date.dayofyear, rng)
        is_weekend = date.dayofweek >= 5
        mean = expected_sales(temp, is_weekend)
        sales = max(0, round(mean * rng.normal(1.0, 0.10)))
        rows.append(
            {
                "date": date.date().isoformat(),
                "weekday": date.dayofweek,      # 0=月 ... 6=日
                "is_weekend": int(is_weekend),
                "max_temp": temp,
                "sales": sales,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-12-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="sales.csv")
    args = parser.parse_args()

    df = generate(args.start, args.end, args.seed)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} 日分を {args.out} に書き出しました")
