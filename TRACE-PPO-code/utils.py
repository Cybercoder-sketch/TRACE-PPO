import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal
from collections import deque
from datetime import datetime


from datetime import datetime
from config import GLOBAL_SEED
# =====================================================
# 1. Logger
# =====================================================
def setup_logger():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(os.getcwd(), f"cicapt_training_seed{GLOBAL_SEED}_unsup_{ts}.log")

    logger = logging.getLogger("TRACE_PPO_CICAPT_UNSUP")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
        ch = logging.StreamHandler(sys.stdout)

        fmt = logging.Formatter("%(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


logger = setup_logger()


def get_training_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"[Device] CUDA enabled: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("[Device] CUDA not available, fallback to CPU.")

    return device


# =====================================================
# 2. Data Utilities
# =====================================================
def load_feature_matrix(path, name, expected_dim=85):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Data] Missing {name} file: {path}")

    arr = np.load(path)

    if arr.ndim != 2:
        raise ValueError(
            f"[Data] {name} should be a 2D feature matrix, "
            f"but got shape={arr.shape}"
        )

    if arr.shape[1] != expected_dim:
        raise ValueError(
            f"[Data] {name} feature dim mismatch: "
            f"expected {expected_dim}, got {arr.shape[1]}, shape={arr.shape}"
        )

    arr = arr.astype(np.float32)

    if not np.isfinite(arr).all():
        logger.info(f"[Data] {name} contains NaN/Inf, applying nan_to_num.")
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    logger.info(f"[Data] Loaded {name}: shape={arr.shape}, dtype={arr.dtype}")
    return arr
# =====================================================
# 4. Survivability + CVaR
# =====================================================
class SurvivabilityMetric:
    def __init__(self, base_weights, beta=0.587, rebound=0.1):
        self.w = np.array(base_weights, dtype=np.float32)
        self.beta = beta
        self.rebound = rebound
        self.tau = np.zeros_like(self.w)

    def update_tau(self, migration_signal):
        for i, s in enumerate(migration_signal):
            if s > 0:
                self.tau[i] = min(1.0, self.tau[i] + s)
            else:
                self.tau[i] = max(0.0, self.tau[i] - self.rebound)

    def survivability(self, s_phy):
        return float(np.dot(self.w * np.exp(-self.beta * self.tau), s_phy))


def weighted_quantile(values, quantiles, sample_weight):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.atleast_1d(quantiles).astype(np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)

    if values.size == 0:
        return np.zeros_like(quantiles, dtype=np.float64)

    if np.sum(sample_weight) <= 1e-12:
        sample_weight = np.ones_like(values, dtype=np.float64)

    sorter = np.argsort(values)

    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_cdf = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_cdf /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_cdf, values)


def empirical_cvar_weighted(costs, weights=None, alpha=0.85, safety_cap_mult=1.5):
    costs = np.asarray(costs, dtype=np.float64)
    n = len(costs)

    if n == 0:
        return 0.0

    if weights is None:
        weights = np.ones(n, dtype=np.float64)

    weights = np.asarray(weights, dtype=np.float64)

    if np.sum(weights) <= 1e-12:
        return float(np.mean(costs))

    if n < 8:
        return float(np.average(costs, weights=weights))

    var = float(weighted_quantile(costs, alpha, weights)[0])
    mask = costs > var

    if not mask.any():
        sorter = np.argsort(costs)[::-1]
        k = max(1, int((1.0 - alpha) * n))
        top_idx = sorter[:k]
        cvar = float(np.average(costs[top_idx], weights=weights[top_idx]))
    else:
        tail_vals = costs[mask]
        tail_w = weights[mask]
        cvar = float(np.average(tail_vals, weights=tail_w))

    cap = safety_cap_mult * float(np.max(costs))
    return min(cvar, cap)


class PIDLagrangian:
    def __init__(
        self,
        kp=0.5,
        ki=0.02,
        kd=0.3,
        lam_init=0.0,
        lam_max=30.0,
        leak=0.985,
        rate_limit=2.0,
        dead_band_frac=0.10,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lam = lam_init
        self.lam_max = lam_max
        self.leak = leak
        self.integral = 0.0
        self.prev_err = 0.0
        self.rate_limit = rate_limit
        self.dead_band_frac = dead_band_frac

    def update(self, cvar, limit):
        if not np.isfinite(cvar):
            return self.lam

        err = cvar - limit
        scale = 1.0 / (1.0 + abs(err) / max(1.0, abs(limit)))

        if abs(err) < self.dead_band_frac * abs(limit):
            self.integral *= 0.95

        not_sat_up = (self.lam < self.lam_max - 1e-3) or (err < 0)
        not_sat_down = (self.lam > 1e-3) or (err > 0)

        if (err > 0 and not_sat_up) or (err < 0 and not_sat_down):
            self.integral = max(-1e3, min(self.integral + err, 1e4))

        if self.lam >= 0.95 * self.lam_max:
            self.integral *= 0.9

        delta = (self.kp * err + self.ki * self.integral + self.kd * (err - self.prev_err)) * scale
        delta = float(np.clip(delta, -self.rate_limit, self.rate_limit))

        if err < 0:
            self.lam *= self.leak

        self.lam = float(np.clip(self.lam + delta, 0.0, self.lam_max))
        self.prev_err = err

        return self.lam

    def clear(self, hard=False):
        if hard:
            self.lam = 0.0
            self.integral = 0.0
            self.prev_err = 0.0
        else:
            self.lam *= 0.3
            self.integral *= 0.2

