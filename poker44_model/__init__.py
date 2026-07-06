"""Participant-owned model package for the Poker44 miner — novel_batchz_c2.

model_name: poker-batchz-c2. Same 180 sanitization-invariant C2 features and the
same ExtraTrees + HistGradientBoosting soft-vote ensemble, wrapped with
per-served-batch STANDARDIZATION (batch-z) at both train and inference: each
feature column is z-scored over the chunks of the served batch (unsupervised,
label-free, no fitted covariance), which removes the benchmark->live LEVEL shift
that collapses C2's live score spread. Trained on benchmark hands sanitized via
prepare_hand_for_miner and z-scored within synthetic per-date batches (train==
serve). Inference does NOT re-sanitize. Output = within-batch rank. n_jobs=1.
See detector.py (inference), features.py (extraction), train_model.py (training),
model.joblib (trained model).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
