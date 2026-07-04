"""Poker44 bot detector — v8-stack (model_name poker-v8-stack).

Domain adaptation strategy (identical DA fit to the deployed CORAL/quantstack
v6/v7, upgraded model + companions):

  TRAINING (baked, offline):
    CORAL covariance alignment of the sanitized benchmark features onto the
    UNLABELED live feature covariance, then a per-feature monotone QUANTILE map
    onto the live marginals, then a per-column z-norm to the live reference,
    then a KS-stable feature mask. The 5-learner OOF-stacked ensemble
    (LightGBM + XGBoost + CatBoost + ExtraTrees(n_jobs=1) + RandomForest(n_jobs=1)
    base bank -> LogisticRegression meta on out-of-fold base preds, with FOCAL
    hard-bot meta reweighting) is fit in that aligned space. No live LABELS are
    ever used.

  INFERENCE (this file):
    Live chunks arrive ALREADY sanitized (prepare_hand_for_miner runs
    validator-side) AND already in the live feature space the model was aligned
    to, so CORAL / quantile map are NOT re-applied here (doing so would
    double-shift already-live-space data). The ONLY per-request transform is the
    aceguard-style per-batch z-norm: re-standardize each feature column across
    the whole query batch, then apply the baked KS mask and the OOF stack. The
    inference score = META_BLEND*meta + (1-META_BLEND)*base_mean; the base-mean
    blend restores the score spread the LogisticRegression meta squashes, and is
    monotone/ranking-neutral for the OOF stack (dup-corr flat across the blend).

  Output = within-batch rank in [0,1] (higher = more bot-like), matching the
  validator's ranking-based reward. ET/RF n_jobs=1 (deterministic, single
  thread). Single-chunk requests fall back to the baked live z-norm reference so
  score_chunk stays consistent with the population.
"""
from __future__ import annotations

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

_MODEL = None
_TX = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(os.path.join(os.path.dirname(__file__),
                                          "model.joblib"))
    return _MODEL


def _tx():
    global _TX
    if _TX is None:
        _TX = np.load(os.path.join(os.path.dirname(__file__),
                                   "v8_full_transform.npz"), allow_pickle=True)
    return _TX


def _rank_normalize(vals):
    n = len(vals)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = round(pos / (n - 1), 6)
    return out


def _featurize(chunks):
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return np.asarray(rows, dtype=float)


def _raw_scores(chunks):
    tx = _tx()
    keep = tx["keep"]
    blend = float(tx["meta_blend"]) if "meta_blend" in tx.files else 0.5
    perbatch = ("znorm_mode" in tx.files
                and str(tx["znorm_mode"]) == "perbatch")
    X = _featurize(chunks)
    if perbatch and X.shape[0] > 1:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)   # aceguard per-batch z-norm
    else:
        mu, sd = tx["znorm_mu"], tx["znorm_sd"]   # single-chunk fallback
    X = (X - mu) / sd
    X = X[:, keep]
    m = _model()
    P = np.column_stack([m["bases"][nm].predict_proba(X)[:, 1]
                         for nm in m["names"]])
    meta = m["meta"].predict_proba(P)[:, 1]
    base_mean = P.mean(axis=1)
    return blend * meta + (1.0 - blend) * base_mean


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk, ranked within the batch."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        return _rank_normalize(list(_raw_scores(chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk score (fallback; batch path is score_batch)."""
    try:
        if not chunk:
            return 0.5
        return round(float(_raw_scores([chunk])[0]), 6)
    except Exception:
        return 0.5
