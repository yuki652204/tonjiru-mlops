#!/bin/sh
# 毎晩の監視パイプライン：監視 →（要対応なら）再学習とPR作成
python monitoring/monitor.py
rc=$?
if [ "$rc" -eq 1 ]; then
  echo "--- 要対応のため、再学習を始めます ---"
  python pipeline/retrain.py
  echo "--- 再学習の終了コード: $? ---"
fi
# Job の成否は監視の結果で決める（要対応なら失敗として記録を残す）
exit $rc
