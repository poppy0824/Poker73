"""Participant-owned model package for the Poker44 miner — v8-stack.

model_name: poker-v8-stack

Bot detector = 5-learner OOF-STACKED ensemble (LightGBM + XGBoost + CatBoost +
ExtraTrees(n_jobs=1) + RandomForest(n_jobs=1) base bank -> LogisticRegression
meta on out-of-fold base predictions, with focal hard-bot meta reweighting) over
a 184-feature sanitization-invariant behavioral feature set (180 CORAL/quantstack
columns + 4 travis coarse-bb-bucket duplication signatures).

Domain adaptation (no live labels): benchmark-train features are CORAL-aligned
(mean + covariance) to the UNLABELED live feature covariance, then quantile-mapped
onto the live marginals, then z-normed and KS-masked, all baked into TRAINING.
Inference does NOT re-sanitize (validator sanitizes) and does NOT re-apply the
CORAL / quantile transform (live is already in the aligned space); the only
per-request transform is an aceguard-style per-batch z-norm. Scores are
within-batch ranks. See detector.py, features.py, model.joblib,
v8_full_transform.npz.
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
