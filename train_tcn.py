import argparse
import json
import os
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class KillZoneSpec:
    start_hhmm: str
    end_hhmm: str

    def _parse(self, s: str) -> time:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))

    @property
    def start(self) -> time:
        return self._parse(self.start_hhmm)

    @property
    def end(self) -> time:
        return self._parse(self.end_hhmm)


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def read_mt5_csv(path: Path, time_column_candidates: Sequence[str], tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = _find_column(df, time_column_candidates)
    if time_col is None:
        raise ValueError(f"Could not find time column in {path}. Columns={list(df.columns)}")

    def pick(col_names: Sequence[str]) -> str:
        col = _find_column(df, col_names)
        if col is None:
            raise ValueError(f"Missing required column among {col_names} in {path}. Columns={list(df.columns)}")
        return col

    o = pick(["open", "Open"])
    h = pick(["high", "High"])
    l = pick(["low", "Low"])
    c = pick(["close", "Close"])

    out = df[[time_col, o, h, l, c]].copy()
    out.columns = ["time", "open", "high", "low", "close"]

    out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=False)
    if out["time"].isna().any():
        bad = int(out["time"].isna().sum())
        raise ValueError(f"Failed to parse {bad} timestamps in {path}")

    if out["time"].dt.tz is None:
        out["time"] = out["time"].dt.tz_localize(tz)
    else:
        out["time"] = out["time"].dt.tz_convert(tz)

    out = out.sort_values("time").reset_index(drop=True)
    return out


def log_returns(close: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    lr = np.zeros_like(close, dtype=np.float64)
    lr[1:] = np.log((close[1:] + eps) / (close[:-1] + eps))
    return lr


def kama(close: np.ndarray, length: int = 21, fast: int = 2, slow: int = 30) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average.
    Standard definition with Efficiency Ratio (ER) and smoothing constant (SC).
    """
    close = np.asarray(close, dtype=np.float64)
    n = close.shape[0]
    out = np.zeros(n, dtype=np.float64)
    out[0] = close[0]

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)

    for i in range(1, n):
        if i < length:
            out[i] = close[i]
            continue
        change = abs(close[i] - close[i - length])
        volatility = np.sum(np.abs(np.diff(close[i - length : i + 1])))
        er = 0.0 if volatility == 0.0 else (change / volatility)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (close[i] - out[i - 1])
    return out


def kama_slope(kama_values: np.ndarray, slope_lookback: int) -> np.ndarray:
    k = np.asarray(kama_values, dtype=np.float64)
    out = np.zeros_like(k, dtype=np.float64)
    if slope_lookback <= 0:
        return out
    out[slope_lookback:] = (k[slope_lookback:] - k[:-slope_lookback]) / float(slope_lookback)
    return out


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def kill_zone_sin_cos(ts: pd.Series, zone: KillZoneSpec, market_tz: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - in_zone (0/1)
      - sin_phase
      - cos_phase

    Phase is time within the kill zone mapped to [0, 2pi). Outside zone -> 0s.
    """
    tlocal = ts.dt.tz_convert(market_tz)
    minutes = (tlocal.dt.hour * 60 + tlocal.dt.minute).to_numpy(np.int32)

    start_m = _time_to_minutes(zone.start)
    end_m = _time_to_minutes(zone.end)

    if end_m > start_m:
        in_zone = (minutes >= start_m) & (minutes < end_m)
        pos = (minutes - start_m).astype(np.float64)
        length = float(end_m - start_m)
    else:
        # Wraps midnight
        in_zone = (minutes >= start_m) | (minutes < end_m)
        pos = np.where(minutes >= start_m, minutes - start_m, minutes + (1440 - start_m)).astype(np.float64)
        length = float((1440 - start_m) + end_m)

    phase = np.zeros_like(pos, dtype=np.float64)
    phase[in_zone] = (pos[in_zone] / max(length, 1.0)) * (2.0 * np.pi)
    sinv = np.zeros_like(phase)
    cosv = np.zeros_like(phase)
    sinv[in_zone] = np.sin(phase[in_zone])
    cosv[in_zone] = np.cos(phase[in_zone])
    return in_zone.astype(np.float64), sinv, cosv


def build_features(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, List[str]]:
    feats: Dict[str, np.ndarray] = {}
    close = df["close"].to_numpy(np.float64)

    if cfg["features"].get("include_ohlc_features", True):
        # Normalize scale somewhat via log price and intrabar ranges.
        o = df["open"].to_numpy(np.float64)
        h = df["high"].to_numpy(np.float64)
        l = df["low"].to_numpy(np.float64)
        eps = 1e-12
        feats["log_close"] = np.log(close + eps)
        feats["hl_range"] = (h - l) / (close + eps)
        feats["oc_return"] = (close - o) / (o + eps)

    if cfg["features"].get("use_log_returns", True):
        feats["logret_1"] = log_returns(close)

    kama_cfg = cfg["features"].get("kama", {})
    if kama_cfg.get("enabled", True):
        k = kama(close, length=int(kama_cfg.get("length", 21)), fast=int(kama_cfg.get("fast", 2)), slow=int(kama_cfg.get("slow", 30)))
        feats["kama"] = k
        slb = int(kama_cfg.get("slope_lookback", 5))
        feats["kama_slope"] = kama_slope(k, slope_lookback=slb)

    kz_cfg = cfg["features"].get("kill_zones", {})
    if kz_cfg.get("enabled", True):
        market_tz = kz_cfg.get("market_tz", "America/New_York")
        london = kz_cfg.get("london", {"start": "02:00", "end": "05:00"})
        ny = kz_cfg.get("newyork", {"start": "08:00", "end": "11:00"})
        in_l, sin_l, cos_l = kill_zone_sin_cos(df["time"], KillZoneSpec(london["start"], london["end"]), market_tz)
        in_n, sin_n, cos_n = kill_zone_sin_cos(df["time"], KillZoneSpec(ny["start"], ny["end"]), market_tz)
        feats["lz_in"] = in_l
        feats["lz_sin"] = sin_l
        feats["lz_cos"] = cos_l
        feats["ny_in"] = in_n
        feats["ny_sin"] = sin_n
        feats["ny_cos"] = cos_n

    # Multi-timeframe master trend: compute M15 close series, run KAMA, take slope, align to M1 bars.
    mt_cfg = cfg["features"].get("master_trend", {})
    if mt_cfg.get("enabled", True):
        tf = str(mt_cfg.get("timeframe", "15T"))
        # Resample to completed bars using last close in interval.
        m15 = (
            df[["time", "close"]]
            .set_index("time")
            .sort_index()
            .resample(tf, label="right", closed="right")
            .last()
            .dropna()
        )
        m15_close = m15["close"].to_numpy(np.float64)
        m15_k = kama(
            m15_close,
            length=int(mt_cfg.get("kama_length", 21)),
            fast=int(mt_cfg.get("fast", 2)),
            slow=int(mt_cfg.get("slow", 30)),
        )
        m15_sl = kama_slope(m15_k, slope_lookback=int(mt_cfg.get("slope_lookback", 3)))
        m15_feat = pd.DataFrame({"time": m15.index, "m15_kama_slope": m15_sl}).sort_values("time")

        # Align each M1 timestamp to last completed M15 bar (asof merge prevents lookahead).
        base = pd.DataFrame({"time": df["time"]}).sort_values("time")
        aligned = pd.merge_asof(base, m15_feat, on="time", direction="backward")
        feats["m15_kama_slope"] = aligned["m15_kama_slope"].to_numpy(np.float64)

    feat_df = pd.DataFrame(feats)

    # Trim initial unstable region (KAMA warmup, returns)
    warmup = 1
    if kama_cfg.get("enabled", True):
        warmup = max(warmup, int(kama_cfg.get("length", 21)))
        warmup = max(warmup, int(kama_cfg.get("slope_lookback", 5)))
    if mt_cfg.get("enabled", True):
        warmup = max(warmup, int(mt_cfg.get("kama_length", 21)))
        warmup = max(warmup, int(mt_cfg.get("slope_lookback", 3)))

    feat_df["time"] = df["time"]
    feat_df = feat_df.iloc[warmup:].reset_index(drop=True)
    return feat_df, [c for c in feat_df.columns if c != "time"]


def forward_return_labels(close: np.ndarray, horizon: int, thr: float, class_order: List[str]) -> np.ndarray:
    """
    Returns integer labels aligned with 'current' index:
      uses forward log return over horizon bars: log(close[t+h]/close[t]).
    """
    close = np.asarray(close, dtype=np.float64)
    n = close.shape[0]
    y = np.full(n, fill_value=class_order.index("NEUTRAL"), dtype=np.int32)
    eps = 1e-12
    fwd = np.zeros(n, dtype=np.float64)
    valid = np.arange(0, n - horizon)
    fwd[valid] = np.log((close[valid + horizon] + eps) / (close[valid] + eps))

    buy_idx = class_order.index("BUY")
    sell_idx = class_order.index("SELL")
    neu_idx = class_order.index("NEUTRAL")
    y[:] = neu_idx
    y[fwd >= thr] = buy_idx
    y[fwd <= -thr] = sell_idx

    # last horizon points have incomplete forward window -> mark neutral but we'll drop later in windowing
    return y


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Windowed samples:
      Xw[i] = X[t-lookback+1 : t+1]
      yw[i] = y[t]
    Drop last horizon to avoid label leakage/incomplete forward.
    """
    n = X.shape[0]
    t_start = lookback - 1
    t_end = n - horizon - 1
    idx = np.arange(t_start, t_end + 1, stride, dtype=np.int32)

    Xw = np.empty((idx.shape[0], lookback, X.shape[1]), dtype=np.float32)
    yw = np.empty((idx.shape[0],), dtype=np.int32)
    for i, t in enumerate(idx):
        Xw[i] = X[t - lookback + 1 : t + 1]
        yw[i] = y[t]
    return Xw, yw


def chronological_split(n: int, train: float, val: float, test: float) -> Tuple[slice, slice, slice]:
    if not np.isclose(train + val + test, 1.0):
        raise ValueError("split ratios must sum to 1.0")
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_train + n_val + n_test)


def compute_class_weights(y: np.ndarray, num_classes: int) -> Dict[int, float]:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = {}
    for i in range(num_classes):
        weights[i] = (total / (num_classes * max(counts[i], 1.0)))
    return weights


def tcn_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout: float,
    use_batch_norm: bool,
    activation: str,
    name: str,
) -> tf.Tensor:
    shortcut = x

    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        use_bias=not use_batch_norm,
        name=f"{name}_conv1",
    )(x)
    if use_batch_norm:
        y = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = tf.keras.layers.Activation(activation, name=f"{name}_act1")(y)
    y = tf.keras.layers.SpatialDropout1D(dropout, name=f"{name}_drop1")(y)

    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        use_bias=not use_batch_norm,
        name=f"{name}_conv2",
    )(y)
    if use_batch_norm:
        y = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = tf.keras.layers.Activation(activation, name=f"{name}_act2")(y)
    y = tf.keras.layers.SpatialDropout1D(dropout, name=f"{name}_drop2")(y)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same", name=f"{name}_proj")(shortcut)

    out = tf.keras.layers.Add(name=f"{name}_add")([shortcut, y])
    out = tf.keras.layers.Activation(activation, name=f"{name}_out")(out)
    return out


def build_tcn_model(
    lookback: int,
    num_features: int,
    num_classes: int,
    cfg: dict,
) -> tf.keras.Model:
    tcn_cfg = cfg["model"]["tcn"]
    head_cfg = cfg["model"]["head"]

    inputs = tf.keras.Input(shape=(lookback, num_features), name="features")
    x = inputs
    filters = int(tcn_cfg.get("filters", 64))
    kernel_size = int(tcn_cfg.get("kernel_size", 3))
    dropout = float(tcn_cfg.get("dropout", 0.1))
    use_bn = bool(tcn_cfg.get("use_batch_norm", True))
    activation = str(tcn_cfg.get("activation", "relu"))
    num_blocks = int(tcn_cfg.get("num_blocks", 6))

    for b in range(num_blocks):
        dilation = 2 ** b
        x = tcn_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout=dropout,
            use_batch_norm=use_bn,
            activation=activation,
            name=f"tcn_b{b}",
        )

    # Causal summary: use last time step representation (no leakage)
    x = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="last_step")(x)
    x = tf.keras.layers.Dense(int(head_cfg.get("dense_units", 64)), activation=activation, name="head_dense")(x)
    x = tf.keras.layers.Dropout(float(head_cfg.get("dropout", 0.2)), name="head_drop")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="tcn_scalper")


def export_onnx_opset13(model: tf.keras.Model, onnx_path: Path, dynamic_batch: bool, opset: int = 13) -> None:
    """
    Exports ONNX in inference mode (BatchNorm frozen) using tf2onnx.
    """
    import tf2onnx

    model.trainable = False
    _ = model(tf.zeros((1,) + tuple(model.input_shape[1:]), dtype=tf.float32), training=False)

    spec = (tf.TensorSpec((None,) + tuple(model.input_shape[1:]), tf.float32, name="features"),)
    if not dynamic_batch:
        spec = (tf.TensorSpec((1,) + tuple(model.input_shape[1:]), tf.float32, name="features"),)

    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=opset,
        output_path=str(onnx_path),
    )
    if onnx_model is None:
        raise RuntimeError("tf2onnx export failed (onnx_model is None)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/train_tcn.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["run"].get("seed", 42))
    set_all_seeds(seed)

    out_root = Path(cfg["run"].get("out_dir", "artifacts"))
    run_name = cfg["run"].get("run_name", "run")
    run_id = f"{run_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = out_root / run_id
    ensure_dir(run_dir)

    raw_glob = cfg["data"]["raw_glob"]
    raw_files = sorted(Path(".").glob(raw_glob))
    if not raw_files:
        raise FileNotFoundError(f"No CSV files found for glob: {raw_glob}")

    tz = cfg["data"].get("tz", "UTC")
    time_candidates = cfg["data"].get("time_column_candidates", ["time", "Time"])

    frames = []
    for fp in raw_files:
        frames.append(read_mt5_csv(fp, time_candidates, tz))
    df = pd.concat(frames, axis=0).sort_values("time").reset_index(drop=True)

    feat_df, feature_cols = build_features(df, cfg)
    close_aligned = df["close"].iloc[(len(df) - len(feat_df)) :].to_numpy(np.float64)

    lookback = int(cfg["data"]["lookback"])
    horizon = int(cfg["data"]["horizon"])
    stride = int(cfg["data"].get("stride", 1))

    labels_cfg = cfg["labels"]
    class_order = list(labels_cfg.get("class_order", ["SELL", "NEUTRAL", "BUY"]))
    if sorted(class_order) != sorted(["SELL", "NEUTRAL", "BUY"]):
        raise ValueError("class_order must contain SELL, NEUTRAL, BUY exactly once")

    thr = float(labels_cfg.get("forward_return_threshold", 0.0006))
    y = forward_return_labels(close_aligned, horizon=horizon, thr=thr, class_order=class_order)

    # Hard master-trend gate (training-time) to mirror runtime rule:
    # Only allow BUY labels when M15 KAMA slope is positive.
    if bool(labels_cfg.get("apply_master_trend_buy_filter", False)):
        if "m15_kama_slope" not in feature_cols:
            raise ValueError("apply_master_trend_buy_filter is true but master trend feature is not enabled")
        buy_idx = class_order.index("BUY")
        neu_idx = class_order.index("NEUTRAL")
        m15_slope = feat_df["m15_kama_slope"].to_numpy(np.float64)
        y = y.copy()
        y[(y == buy_idx) & ~(m15_slope > 0.0)] = neu_idx

    X = feat_df[feature_cols].to_numpy(np.float32)
    Xw, yw = make_windows(X, y, lookback=lookback, horizon=horizon, stride=stride)

    split_cfg = cfg["data"]["split"]
    s_train, s_val, s_test = chronological_split(len(Xw), float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]))
    X_train, y_train = Xw[s_train], yw[s_train]
    X_val, y_val = Xw[s_val], yw[s_val]
    X_test, y_test = Xw[s_test], yw[s_test]

    # Fit scaler on training only; apply per-feature across all timesteps.
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_2d)

    def transform(Xseq: np.ndarray) -> np.ndarray:
        X2 = Xseq.reshape(-1, Xseq.shape[-1])
        X2 = scaler.transform(X2)
        return X2.reshape(Xseq.shape).astype(np.float32)

    X_train = transform(X_train)
    X_val = transform(X_val)
    X_test = transform(X_test)

    num_classes = 3
    model = build_tcn_model(lookback=lookback, num_features=X_train.shape[-1], num_classes=num_classes, cfg=cfg)

    lr = float(cfg["train"].get("lr", 8e-4))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(cfg["train"].get("early_stopping_patience", 7)), restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=int(cfg["train"].get("reduce_lr_patience", 3)), factor=0.5, min_lr=1e-6),
    ]

    class_weight = None
    if bool(cfg["train"].get("class_weight", True)):
        class_weight = compute_class_weights(y_train, num_classes=num_classes)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(cfg["train"].get("epochs", 40)),
        batch_size=int(cfg["train"].get("batch_size", 256)),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # Evaluation
    probs = model.predict(X_test, batch_size=1024, verbose=0)
    y_pred = probs.argmax(axis=-1).astype(np.int32)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save artifacts
    keras_path = run_dir / "model.keras"
    model.save(keras_path)

    feature_spec = {
        "feature_cols": feature_cols,
        "lookback": lookback,
        "horizon": horizon,
        "class_order": class_order,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "timezone": tz,
        "config_path": str(args.config),
    }
    with open(run_dir / "feature_spec.json", "w", encoding="utf-8") as f:
        json.dump(feature_spec, f, indent=2)

    metrics = {
        "run_id": run_id,
        "num_samples": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "classification_report": report,
        "confusion_matrix": cm,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Export ONNX opset 13 in inference mode (BN frozen)
    onnx_cfg = cfg.get("export", {})
    onnx_path = run_dir / "model.onnx"
    export_onnx_opset13(
        model=model,
        onnx_path=onnx_path,
        dynamic_batch=bool(onnx_cfg.get("dynamic_batch", True)),
        opset=int(onnx_cfg.get("onnx_opset", 13)),
    )

    print(f"Run complete: {run_id}")
    print(f"Artifacts: {run_dir}")
    print(f"Saved: {keras_path.name}, {onnx_path.name}, feature_spec.json, metrics.json")


if __name__ == "__main__":
    main()

