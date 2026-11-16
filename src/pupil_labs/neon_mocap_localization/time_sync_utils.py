import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view


def align_signals(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    y_ts: npt.NDArray[np.float64],
) -> int:
    if np.isnan(x).any():
        x[np.isnan(x)] = np.nanmean(x)

    if np.isnan(y).any():
        y[np.isnan(y)] = np.nanmean(y)

    n = len(x)
    m = len(y)

    max_corr_idx = 0
    if n <= m:
        windows = sliding_window_view(y, n)
        sq_diff = np.sum((windows - x) ** 2, axis=1)
        max_corr_idx = int(np.argmin(sq_diff))
    else:
        # x is longer than y: slide y across x to find best alignment
        windows = sliding_window_view(x, m)
        sq_diff = np.sum((windows - y) ** 2, axis=1)
        best_offset = int(np.argmin(sq_diff))
        # x starts best_offset samples before y[0], so offset in y-frame is negative
        max_corr_idx = -best_offset

    # dt = np.mean(np.diff(y_ts)) if len(y_ts) > 1 else 1.0
    # x_time_in_y = y_ts[0] + np.arange(max_corr_idx, max_corr_idx + n) * dt
    # x_idxs_in_y = list(range(max_corr_idx, max_corr_idx + n))

    return max_corr_idx
