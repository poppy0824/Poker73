"""Reproducible training for C2-variant `poker-c2-recency` -> model.joblib.

This is C2 (v5_sani) EXACTLY -- same 180 sanitization-invariant behavioral
features, same ExtraTrees + HistGradientBoosting soft-vote ensemble, same
train==serve sanitizer (prepare_hand_for_miner) -- with ONE delta:

  * RECENCY sample-weighting. Chunks whose benchmark date falls in the most
    recent `--recent-days` (default 12) calendar dates get a sample_weight of
    `--recent-weight` (default 1.75); all older chunks keep weight 1.0.

Rationale (population prior, NOT a capture fit): the sanitized action-mix on the
most recent benchmark dates drifts toward the live distribution (fold-heavy ->
call/check-heavy). Up-weighting those dates nudges the decision surface toward
the live population WITHOUT ever looking at a single live chunk or fitting any
capture-derived DA transform. ALL dates are kept -- nothing is dropped.

The delta produces a genuinely distinct fitted artifact (different sample
weights -> different tree splits / gradient steps -> different model.joblib),
so it is safe as a separate UID entry for copy-DQ purposes.

All learners run single-threaded (n_jobs=1 / default) for determinism.

    python3 poker44_model/train_model.py \
        --data /root/ares/Poker/train/raw \
        --payload-view /root/ares/Poker/main/poker44/validator/payload_view.py
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import typing

import numpy as np
import joblib
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              VotingClassifier)

from poker44_model.features import chunk_features, FEATURE_NAMES

_DATE_RE = re.compile(r"chunks_(\d{4}-\d{2}-\d{2})\.json$")


def _load_sanitizer(pv_path):
    spec = importlib.util.spec_from_file_location("_p44_payload_view", pv_path)
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    pv.Optional = typing.Optional  # payload_view uses Optional but never imports it
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
    """Yield (chunk, label, date) so we can date-weight per chunk."""
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "chunks_*.json"))):
        m = _DATE_RE.search(os.path.basename(f))
        date = m.group(1) if m else ""
        for rc in json.load(open(f)).get("chunks", []):
            for g, l in zip(rc.get("chunks") or [], rc.get("groundTruth") or []):
                out.append((g, int(l), date))
    return out


def build_ensemble(seed=0):
    # n_jobs=1: single-threaded / deterministic (variant contract).
    et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4,
                              random_state=seed, n_jobs=1)
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03,
                                         max_iter=300, l2_regularization=1.0,
                                         random_state=seed)
    return VotingClassifier(estimators=[("et", et), ("hgb", hgb)], voting="soft")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to train/raw chunk JSON dir")
    ap.add_argument("--payload-view", required=True,
                    help="path to poker44/validator/payload_view.py (the sanitizer)")
    ap.add_argument("--recent-days", type=int, default=12,
                    help="number of most-recent calendar dates to up-weight")
    ap.add_argument("--recent-weight", type=float, default=1.25,
                    help="sample_weight for chunks on the recent dates (older=1.0). "
                         "1.25 x last-12-dates was the OOS sweep winner (dup-corr "
                         ">= C2, gentle enough to preserve ranking; heavier weights "
                         "degraded OOS dup-corr).")
    args = ap.parse_args()

    sanitize_chunk = _load_sanitizer(args.payload_view)

    data = load(args.data)
    all_dates = sorted({d for (_, _, d) in data if d})
    recent = set(all_dates[-args.recent_days:]) if args.recent_days > 0 else set()

    rows, y, w = [], [], []
    for g, l, date in data:
        feats = chunk_features(sanitize_chunk(g))   # TRAIN == SERVE: sanitize raw hands
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
        y.append(l)
        w.append(args.recent_weight if date in recent else 1.0)
    X = np.array(rows, dtype=float)
    y = np.array(y)
    w = np.array(w, dtype=float)

    model = build_ensemble(seed=0).fit(X, y, sample_weight=w)

    out = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, out)
    n_recent = int((w != 1.0).sum())
    print(f"wrote {out} ({len(data)} examples, {len(FEATURE_NAMES)} features)")
    print(f"  recency: up-weighted {n_recent}/{len(data)} chunks "
          f"(dates {sorted(recent)[0]}..{sorted(recent)[-1]}) x{args.recent_weight}")


if __name__ == "__main__":
    main()
