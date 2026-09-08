import math
from typing import Any, Dict, List

import numpy as np


def is_numeric_sequence(v: Any) -> bool:
    if not isinstance(v, list) or len(v) == 0:
        return False
    for x in v:
        if x is None:
            return False
        if not isinstance(x, (int, float)):
            return False
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return False
    return True


def get_common_epoch_metrics(logs: List[Dict[str, Any]], preferred_metrics: List[str]) -> List[str]:
    common = []
    for key in preferred_metrics:
        ok = True
        for log in logs:
            if key not in log or not is_numeric_sequence(log.get(key)):
                ok = False
                break
        if ok:
            common.append(key)
    return common


def to_epoch_axis(log: Dict[str, Any], metric_len: int) -> np.ndarray:
    epoch = log.get("epoch")
    if isinstance(epoch, list) and len(epoch) == metric_len and all(
        isinstance(x, (int, float)) for x in epoch
    ):
        return np.array(epoch, dtype=float)
    return np.arange(metric_len, dtype=float)


def sanitize_filename(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size < window:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    y_pad = np.pad(y, (window // 2, window - 1 - (window // 2)), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")
