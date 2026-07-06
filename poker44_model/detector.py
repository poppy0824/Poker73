"""Poker44 bot detector — novel_batchz_c2 (model_name poker-batchz-c2).

Same 180 sanitization-invariant C2 features (features.py) and the same
ExtraTrees+HistGradientBoosting soft-vote ensemble, but wrapped with
**per-served-batch STANDARDIZATION** (batch-z) at BOTH train and inference.

Motivation. The 2026-07-06 eval sanitization plus the benchmark->live population
gap shift the ABSOLUTE feature levels (raise_sh_*/action-share/BB aggregates drift
0.68-0.93 sd), which pushes live chunks out of the trained feature range and
COLLAPSES the raw predict_proba spread (C2 live raw-STD ~0.065 -> near-random
ranking). Batch-z removes that per-batch LEVEL shift unsupervised, label-free and
without any fitted covariance: for each feature column, z = (x - batch_mean) /
(batch_std + eps), clipped to +/-5, computed over the chunks in the SINGLE served
batch. Training applies the identical z within synthetic per-date batches so the
model sees the same standardized distribution it will see live (train==serve).
This neutralizes the drift instead of dropping the signal (unlike robust_c2) and
un-collapses the live spread (raw-STD ~0.16, new-eval dup-corr +0.30 -> +0.56).

Inference does NOT sanitize (live chunks arrive already sanitized by the
validator). It featurizes the incoming chunks, batch-z's the batch, predicts, and
returns within-batch rank. n_jobs=1 (batched predict must not deadlock).
"""
from __future__ import annotations

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

_MODEL = None
_EPS = 1e-6
_MIN_BATCH = 10   # below this, feature std is unreliable -> neutral fallback


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
    return _MODEL


def _rank_normalize(vals):
    n = len(vals)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = round(pos / (n - 1), 6)
    return out


def _batch_z(X):
    """Column z-score across the chunks of THIS served batch, clip +/-5."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return np.clip((X - mu) / (sd + _EPS), -5.0, 5.0)


def _feature_matrix(chunks):
    rows = []
    for c in chunks:
        feats = chunk_features(c)          # C2's 180 sanitization-invariant features
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return np.array(rows, dtype=float)


def _raw_scores(model, chunks):
    X = _feature_matrix(chunks)
    Xz = _batch_z(X)                        # batch-z: unsupervised level alignment
    return model.predict_proba(Xz)[:, 1]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk, batch-z'd then ranked within batch."""
    chunks = chunks or []
    if not chunks:
        return []
    # Batch-z needs several chunks to estimate per-feature mean/std reliably.
    if len(chunks) < _MIN_BATCH:
        return [0.5] * len(chunks)
    try:
        return _rank_normalize(list(_raw_scores(_model(), chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk score. Batch-z is undefined for one chunk -> neutral."""
    try:
        if not chunk:
            return 0.5
        return 0.5
    except Exception:
        return 0.5
