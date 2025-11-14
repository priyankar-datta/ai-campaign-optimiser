#!/usr/bin/env bash
set -euxo pipefail

cd /workspace

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

python3 data/generate_synthetic_data.py --rows 2000 --out data/synthetic_ads.csv
python3 train/train_xgb.py --data data/synthetic_ads.csv --model_out models/ctr_xgb.json

echo "Setup complete!"
