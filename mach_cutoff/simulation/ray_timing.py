"""Derive path length and acoustic travel time along integrated ray trajectories."""

from __future__ import annotations

import numpy as np


def path_metrics_from_trajectory(
    traj_local_m: np.ndarray,
    *,
    field=None,
    reference_speed_mps: float = 340.29,
    fallback_c_eff_mps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (path_s_m, path_t_rel_s, path_c_eff_mps) for each sample.

    Travel time uses ``dt = ds / c_eff`` with ``c_eff = reference_speed / n`` when a
    refractive-index field is available; otherwise ``fallback_c_eff_mps`` (or the
    reference sound speed) is used as a constant.
    """
    traj = np.asarray(traj_local_m, dtype=float)
    if traj.ndim != 2 or traj.shape[1] < 3 or traj.shape[0] == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty

    n_pts = int(traj.shape[0])
    path_s = np.zeros(n_pts, dtype=np.float64)
    path_t = np.zeros(n_pts, dtype=np.float64)
    c_eff = np.zeros(n_pts, dtype=np.float64)

    ref = max(float(reference_speed_mps), 1e-6)
    fallback = float(fallback_c_eff_mps) if fallback_c_eff_mps is not None else ref
    fallback = max(fallback, 1e-6)

    n_and_grad = getattr(field, "n_and_grad", None) if field is not None else None
    for i in range(n_pts):
        if callable(n_and_grad):
            try:
                n_val, _ = n_and_grad(traj[i, :3])
                n_val = float(n_val)
                if not np.isfinite(n_val) or n_val <= 1e-8:
                    c_eff[i] = fallback
                else:
                    c_eff[i] = ref / n_val
            except Exception:
                c_eff[i] = fallback
        else:
            c_eff[i] = fallback
        if not np.isfinite(c_eff[i]) or c_eff[i] <= 1e-8:
            c_eff[i] = fallback

    for i in range(1, n_pts):
        ds = float(np.linalg.norm(traj[i, :3] - traj[i - 1, :3]))
        if not np.isfinite(ds) or ds < 0.0:
            ds = 0.0
        path_s[i] = path_s[i - 1] + ds
        c_mid = 0.5 * (c_eff[i] + c_eff[i - 1])
        path_t[i] = path_t[i - 1] + ds / max(c_mid, 1e-6)

    return path_s, path_t, c_eff
