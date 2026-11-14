"""
再学習とPR作成：監視ジョブが「要対応」と判定したあとに動く

  1. まだマージされていない再学習のPRがあれば、何もしない（毎晩PRが増えるのを防ぐ）
  2. 候補モデルを「直近7日を使わずに」学習し、その7日で本番モデルと同じ条件で比べる
  3. 候補モデルが本番より良く、かつ基準以内なら、全データで学習し直してMLflowに登録する
  4. GitHubに「MODEL_VERSION を新しい番号に変える」PRを作る（マージは人が行う）

終了コード:
  0 = PRを作った、または未マージのPRがあるので何もしなかった
  3 = 候補モデルが良くならなかった（人の判断が必要）

使い方（手元で試すとき。--dry-run ならGitHubには何もしない）:
  python pipeline/retrain.py --as-of 2026-04-15 --predict-url http://localhost:8000 \
    --tracking-uri http://localhost:5000 --dry-run
"""
import argparse
import base64
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "data-gen"))
sys.path.insert(0, os.path.join(ROOT, "monitoring"))
from generate import generate  # noqa: E402
from monitor import DATA_START, ask_prediction  # noqa: E402

FEATURES = ["max_temp", "is_weekend"]  # training/train.py と同じ
MODEL_NAME = "tonjiru-sales"
MANIFEST_PATH = "deploy/predict-api.yaml"
BRANCH_PREFIX = "retrain-"


# ---------- GitHub ----------
def github(method: str, path: str, token: str, body: dict | None = None) -> dict | list:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as res:
            return json.load(res)
    except urllib.error.HTTPError as e:
        # GitHub が返した理由（例: 権限が足りない）を、そのままログに残す
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} が {e.code} で失敗: {detail}") from None


def open_retrain_prs(repo: str, token: str) -> list:
    pulls = github("GET", f"/repos/{repo}/pulls?state=open&per_page=100", token)
    return [p for p in pulls if p["head"]["ref"].startswith(BRANCH_PREFIX)]


def create_pr(repo: str, token: str, new_version: str, title: str, body: str) -> str:
    branch = f"{BRANCH_PREFIX}v{new_version}"
    main_sha = github("GET", f"/repos/{repo}/git/ref/heads/main", token)["object"]["sha"]
    github("POST", f"/repos/{repo}/git/refs", token, {"ref": f"refs/heads/{branch}", "sha": main_sha})

    current = github("GET", f"/repos/{repo}/contents/{MANIFEST_PATH}?ref=main", token)
    text = base64.b64decode(current["content"]).decode()
    updated = bump_model_version(text, new_version)
    github(
        "PUT",
        f"/repos/{repo}/contents/{MANIFEST_PATH}",
        token,
        {
            "message": title,
            "content": base64.b64encode(updated.encode()).decode(),
            "sha": current["sha"],
            "branch": branch,
        },
    )
    pr = github("POST", f"/repos/{repo}/pulls", token,
                {"title": title, "head": branch, "base": "main", "body": body})
    return pr["html_url"]


def bump_model_version(manifest: str, new_version: str) -> str:
    """マニフェストの MODEL_VERSION の値だけを書き換える"""
    pattern = r'(- name: MODEL_VERSION\n\s+value: )"[^"]*"'
    updated, count = re.subn(pattern, rf'\g<1>"{new_version}"', manifest)
    if count != 1:
        raise RuntimeError("MODEL_VERSION の行が1つだけ見つかりませんでした")
    return updated


# ---------- 比較と学習 ----------
def champion_mae(window: pd.DataFrame, predict_url: str) -> tuple[float, str]:
    preds = [ask_prediction(predict_url, r.date, float(r.max_temp)) for r in window.itertuples()]
    mae = mean_absolute_error(window["sales"], [p["predicted_sales"] for p in preds])
    return mae, preds[0]["model_version"]


def challenger_mae(history: pd.DataFrame, window: pd.DataFrame) -> float:
    """直近7日を使わずに学習した候補モデルで、直近7日を予測する（答えを見ずに答え合わせ）"""
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(history[FEATURES].astype(float), history["sales"])
    return mean_absolute_error(window["sales"], model.predict(window[FEATURES].astype(float)))


def train_and_register(df: pd.DataFrame, as_of: dt.date, tracking_uri: str) -> str:
    """training/train.py を使って全データで学習し、登録されたバージョン番号を返す"""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "sales.csv")
        df.to_csv(csv, index=False)
        train_end = (as_of + dt.timedelta(days=1)).isoformat()  # as_of の日まで含める
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "training", "train.py"),
             "--data", csv, "--train-end", train_end, "--tracking-uri", tracking_uri],
            check=True,
        )
    mlflow.set_tracking_uri(tracking_uri)
    versions = mlflow.MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    return str(max(int(v.version) for v in versions))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=os.environ.get("AS_OF") or dt.date.today().isoformat())
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--mae-threshold", type=float, default=10.0)
    parser.add_argument("--predict-url", default=os.environ.get("PREDICT_URL", "http://predict-api:8000"))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPO", "yuki652204/tonjiru-mlops"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="GitHubには何もせず、作るPRの内容を表示する")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    if not args.dry_run and not token:
        print("GITHUB_TOKEN がありません")
        return 2

    # 1. 未マージの再学習PRがあれば、何もしない
    if not args.dry_run:
        existing = open_retrain_prs(args.repo, token)
        if existing:
            print(f"未マージの再学習PRがあるため、今回は再学習しません: {existing[0]['html_url']}")
            return 0

    # 2. 候補モデルと本番モデルを、直近7日で比べる
    as_of = dt.date.fromisoformat(args.as_of)
    df = generate(DATA_START, as_of.isoformat(), args.seed)
    history, window = df.iloc[: -args.window], df.iloc[-args.window:]

    champ_mae, champ_version = champion_mae(window, args.predict_url)
    chall_mae = challenger_mae(history, window)
    period = f"{window['date'].iloc[0]} 〜 {window['date'].iloc[-1]}"

    print(f"=== 再学習の判定（今日={as_of}、比較期間 {period}） ===")
    print(f"本番モデル（version {champ_version}）の誤差: {champ_mae:.1f} 杯")
    print(f"候補モデル（{history['date'].iloc[-1]} までのデータで学習）の誤差: {chall_mae:.1f} 杯")

    # 3. 候補が本番より良く、かつ基準以内のときだけ先に進む
    if not (chall_mae < champ_mae and chall_mae <= args.mae_threshold):
        print("判定: 候補モデルが十分に良くならないため、PRは作りません（人の判断が必要）")
        return 3

    print("判定: 候補モデルのほうが良い。全データで学習し直して登録します")
    new_version = train_and_register(df, as_of, args.tracking_uri)

    # 4. PRを作る
    title = f"予測モデルを version {new_version} に切り替え（監視ジョブによる自動再学習）"
    body = "\n".join(
        [
            f"監視ジョブが {as_of} に要対応と判定したため、自動で再学習した。",
            "",
            f"| 比較期間 {period} | 誤差（MAE） |",
            "|---|---|",
            f"| 本番 version {champ_version} | {champ_mae:.1f} 杯 |",
            f"| 候補（直近{args.window}日を使わずに学習） | {chall_mae:.1f} 杯 |",
            "",
            f"version {new_version} は、{DATA_START} 〜 {as_of} の全データで学習し直したモデル。",
            "内容を確認して問題なければマージする（マージ後、Argo CD が本番に反映する）。",
        ]
    )

    if args.dry_run:
        print(f"[dry-run] 作るPR: {title}")
        print(body)
        return 0

    url = create_pr(args.repo, token, new_version, title, body)
    print(f"PRを作りました: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
