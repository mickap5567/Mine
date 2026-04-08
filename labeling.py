from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


Label = Literal[-1, 0, 1]  # -1=SELL, 0=NEUTRAL, 1=BUY


@dataclass(frozen=True)
class TripleBarrierResult:
    label: Label
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    upper: float
    lower: float


def triple_barrier_label_one(
    df: pd.DataFrame,
    entry_idx: int,
    *,
    atr_col: str = "atr",
    up_mult: float = 1.5,
    dn_mult: float = 0.8,
    time_limit_minutes: int = 15,
    price_col: str = "close",
    time_col: str = "time",
    high_col: str = "high",
    low_col: str = "low",
    same_bar_policy: Literal["adverse_first", "favorable_first", "neutral"] = "adverse_first",
) -> TripleBarrierResult:
    """
    Triple barrier labeling using OHLC bars (first-touch approximation).

    Buy(1) if High hits P0 + up_mult*ATR0 before Low hits P0 - dn_mult*ATR0 and before time barrier.
    Sell(-1) if Low hits first.
    Neutral(0) if time barrier hits first or data ends.

    same_bar_policy resolves the OHLC ambiguity when both upper & lower are crossed in the same bar.
    """
    if entry_idx < 0 or entry_idx >= len(df) - 1:
        raise ValueError("entry_idx out of range")

    t0 = pd.Timestamp(df.iloc[entry_idx][time_col])
    deadline = t0 + pd.Timedelta(minutes=int(time_limit_minutes))

    p0 = float(df.iloc[entry_idx][price_col])
    atr0 = float(df.iloc[entry_idx][atr_col])
    if not np.isfinite(atr0) or atr0 <= 0:
        return TripleBarrierResult(
            label=0,
            exit_idx=entry_idx,
            exit_time=t0,
            exit_price=p0,
            exit_reason="ATR_INVALID",
            upper=np.nan,
            lower=np.nan,
        )

    upper = p0 + float(up_mult) * atr0
    lower = p0 - float(dn_mult) * atr0

    for i in range(entry_idx + 1, len(df)):
        ti = pd.Timestamp(df.iloc[i][time_col])
        if ti > deadline:
            j = i - 1
            tj = pd.Timestamp(df.iloc[j][time_col])
            pj = float(df.iloc[j][price_col])
            return TripleBarrierResult(
                label=0,
                exit_idx=j,
                exit_time=tj,
                exit_price=pj,
                exit_reason="TIME",
                upper=upper,
                lower=lower,
            )

        hi = float(df.iloc[i][high_col])
        lo = float(df.iloc[i][low_col])
        hit_up = hi >= upper
        hit_dn = lo <= lower

        if hit_up and hit_dn:
            if same_bar_policy == "neutral":
                return TripleBarrierResult(0, i, ti, float(df.iloc[i][price_col]), "BOTH_SAME_BAR_NEUTRAL", upper, lower)
            if same_bar_policy == "favorable_first":
                return TripleBarrierResult(1, i, ti, upper, "BOTH_SAME_BAR_ASSUME_UP_FIRST", upper, lower)
            return TripleBarrierResult(-1, i, ti, lower, "BOTH_SAME_BAR_ASSUME_DN_FIRST", upper, lower)

        if hit_up:
            return TripleBarrierResult(1, i, ti, upper, "UP", upper, lower)
        if hit_dn:
            return TripleBarrierResult(-1, i, ti, lower, "DOWN", upper, lower)

    last = len(df) - 1
    tl = pd.Timestamp(df.iloc[last][time_col])
    pl = float(df.iloc[last][price_col])
    return TripleBarrierResult(0, last, tl, pl, "EOD", upper, lower)


def triple_barrier_labels_vectorized(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr",
    up_mult: float = 1.5,
    dn_mult: float = 0.8,
    time_limit_minutes: int = 15,
    price_col: str = "close",
    time_col: str = "time",
    high_col: str = "high",
    low_col: str = "low",
    same_bar_policy: Literal["adverse_first", "favorable_first", "neutral"] = "adverse_first",
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper: labels every index in [start_idx, end_idx).
    Returns (labels, exit_indices).
    This is intentionally simple (looping); for large datasets prefer batching/numba.
    """
    if end_idx is None:
        end_idx = len(df)
    labels = np.zeros(end_idx - start_idx, dtype=np.int8)
    exits = np.full(end_idx - start_idx, -1, dtype=np.int32)
    for k, i in enumerate(range(start_idx, end_idx)):
        if i >= len(df) - 1:
            labels[k] = 0
            exits[k] = i
            continue
        r = triple_barrier_label_one(
            df,
            i,
            atr_col=atr_col,
            up_mult=up_mult,
            dn_mult=dn_mult,
            time_limit_minutes=time_limit_minutes,
            price_col=price_col,
            time_col=time_col,
            high_col=high_col,
            low_col=low_col,
            same_bar_policy=same_bar_policy,
        )
        labels[k] = int(r.label)
        exits[k] = int(r.exit_idx)
    return labels, exits

