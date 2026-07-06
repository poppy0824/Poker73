"""Reproducible training for novel_batchz_c2 (poker-batchz-c2) -> model.joblib.

Same C2 features + ExtraTrees/HistGradientBoosting ensemble, but every raw
benchmark hand is (1) passed through the validator's prepare_hand_for_miner
sanitizer (train==serve), then (2) the 180-feature rows are grouped into synthetic
per-date "served batches" and column z-scored WITHIN each batch (batch-z), clipped
to +/-5. Inference applies the identical batch-z over the chunks of each live
served batch (detector.py). This removes the benchmark->live absolute-level shift
that collapses C2's live spread, unsupervised and label-free.

    python3 poker44_model/train_model.py --data /root/ares/Poker/train/raw \
        --payload-view /root/ares/Poker/main/poker44/validator/payload_view.py
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import typing

import numpy as np
import joblib
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              VotingClassifier)

from poker44_model.features import chunk_features, FEATURE_NAMES

EPS = 1e-6


def _load_sanitizer(pv_path):
    spec = importlib.util.spec_from_file_location("_p44_payload_view", pv_path)
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    pv.Optional = typing.Optional
    fn = pv.prepare_hand_for_miner

    def sanitize_chunk(chunk):
        out = []
        for h in (chunk or []):
            try:
                out.append(fn(h))
            except Exception:
                out.append(h)
        return out

    return sanitize_chunk


def load(raw):
    """Return list of (group_hands, label, date). Date = synthetic served batch."""
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "chunks_*.json"))):
        date = os.path.basename(f).replace("chunks_", "").replace(".json", "")
        for rc in json.load(open(f)).get("chunks", []):
            for g, l in zip(rc.get("chunks") or [], rc.get("groundTruth") or []):
                out.append((g, int(l), date))
    return out


def batch_z_by_group(X, groups):
    """Column z-score within each group (synthetic served batch), clip +/-5."""
    X = np.asarray(X, float)
    out = np.empty_like(X)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        sub = X[idx]
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0)
        out[idx] = np.clip((sub - mu) / (sd + EPS), -5.0, 5.0)
    return out


def build_ensemble(seed=0):
    et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4,
                              random_state=seed, n_jobs=1)
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03,
                                         max_iter=300, l2_regularization=1.0,
                                         random_state=seed)
    return VotingClassifier(estimators=[("et", et), ("hgb", hgb)], voting="soft")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--payload-view", required=True)
    args = ap.parse_args()

    sanitize_chunk = _load_sanitizer(args.payload_view)

    data = load(args.data)
    rows, y, groups = [], [], []
    for g, l, d in data:
        feats = chunk_features(sanitize_chunk(g))   # TRAIN == SERVE: sanitize raw hands
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
        y.append(l)
        groups.append(d)
    X = np.nan_to_num(np.array(rows, dtype=float))
    y = np.array(y)
    groups = np.array(groups)

    Xz = batch_z_by_group(X, groups)                # batch-z within synthetic served batches

    model = build_ensemble(seed=0).fit(Xz, y)

    out = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, out)
    print(f"wrote {out} ({len(data)} examples, {len(FEATURE_NAMES)} features, "
          f"{len(set(groups.tolist()))} synthetic batches)")


if __name__ == "__main__":
    main()
