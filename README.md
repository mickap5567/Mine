# EA Gold Scalper 2026 — ML Training Blueprint (TCN → ONNX)

This project trains a Temporal Convolutional Network (TCN) on MT5 CSV exports (XAUUSD) and exports an ONNX model (opset 13) suitable for MT5 inference integration.

## What you get
- Feature engineering:
  - Log returns
  - KAMA slope (Kaufman's Adaptive Moving Average)
  - Sin/Cos time encoding for **London** and **New York** kill zones
- Model:
  - Dilated causal **TCN** with residual blocks + BatchNorm
  - Output: 3-class softmax **[Sell, Neutral, Buy]** (configurable)
- Data:
  - Custom loader for MT5 CSV (supports typical OHLCV layouts)
  - Windowed sequences for scalping-style classification
- Export:
  - ONNX export via `tf2onnx` with **opset 13**
  - BatchNorm handled by exporting in inference mode (model frozen)

## Setup

Create a venv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data format (MT5 CSV)
The trainer expects a CSV with at least these columns (case-insensitive):
- `time` (or `Time`, `DATE`, etc.)
- `open`, `high`, `low`, `close`
- optional: `tick_volume` / `volume` / `real_volume`

Put files under:
`data/raw/`

## Train

```bash
python -m training.train_tcn --config configs/train_tcn.yaml
```

Outputs go to:
- `artifacts/<run_id>/model.keras`
- `artifacts/<run_id>/model.onnx`
- `artifacts/<run_id>/feature_spec.json`
- `artifacts/<run_id>/metrics.json`

## Labels (3-class)
Default labeling is forward-return thresholding:
- **Buy** if forward log return ≥ +threshold
- **Sell** if forward log return ≤ -threshold
- **Neutral** otherwise

Tune thresholds per timeframe/spread regime in `configs/train_tcn.yaml`.

## Publish to GitHub

1. Install [Git for Windows](https://git-scm.com/download/win) (or use GitHub Desktop).
2. Create a new empty repository on [GitHub](https://github.com/new) (no README/license if you already have them locally).
3. In this folder:

```powershell
git init
git add .
git commit -m "Initial commit: EA Gold Scalper 2026 training + MQL5"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

Replace `YOUR_USER/YOUR_REPO` with your account and repository name. Use SSH remote URL instead if you prefer.

