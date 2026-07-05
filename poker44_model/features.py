"""v8-stack features = 180-column CORAL/quantstack sanitization-invariant
behavioral feature set (vendored verbatim from the deployed v6_da/coral feature
extractor, see features_coral.py) PLUS 4 travis-style COARSENED amount-BUCKET
duplication signatures.

Rationale (travis / run_dup): the raw sig_amtbucket family in the CORAL feature
set keys hands on bet-size buckets cut at 0.5/1/2/5 bb, thresholds that live in
the BENCHMARK magnitude regime (~36bb pots); on the live feed (~1bb pots) almost
every amount collapses into a single bucket, degenerating the tell. We ADD a
magnitude-tolerant coarse bucket family (<=1,1-2,2-4,4-8,8-16,16-40,>40 bb, the
same structure-preserving buckets as pseudolabel/run_dup.py) and the combined
(action_type, coarse-bucket) signature. These survive the ~36bb->~1bb shift, so
the cross-hand duplication signature transfers to the live population.

The CORAL FEATURE_NAMES order is preserved and the 4 new columns are APPENDED at
the end, so the baked CORAL/quantstack alignment machinery (which is fit
per-column, column-order agnostic) stays valid. Self-contained (stdlib only).
"""
from __future__ import annotations

from collections import Counter

from poker44_model import features_coral as _m

_coral_chunk_features = _m.chunk_features
hand_features = _m.hand_features
_CORAL_NAMES = list(_m.FEATURE_NAMES)
_f = _m._f
_i = _m._i
_div = _m._div


# --- travis-style coarse, magnitude-tolerant bb bucket (== run_dup._bucket) ---
def _coarse_bucket(bb):
    try:
        x = float(bb)
    except (TypeError, ValueError):
        return "na"
    if x <= 0:
        return "0"
    for hi, lab in ((1, "<=1"), (2, "1-2"), (4, "2-4"), (8, "4-8"),
                    (16, "8-16"), (40, "16-40")):
        if x <= hi:
            return lab
    return ">40"


# new columns appended after the CORAL block
_NEW_NAMES = [
    "sig_amtbucketC_top_share", "sig_amtbucketC_unique_share",
    "sig_actbucketC_top_share", "sig_actbucketC_unique_share",
]

FEATURE_NAMES = _CORAL_NAMES + _NEW_NAMES


def chunk_features(chunk):
    out = _coral_chunk_features(chunk)
    if not chunk:
        for k in _NEW_NAMES:
            out[k] = 0.0
        return out
    cbsig, abcsig = [], []
    for h in chunk:
        acts = (h.get("actions") or []) if isinstance(h, dict) else []
        cb = tuple(_coarse_bucket((a or {}).get("normalized_amount_bb"))
                   for a in acts)
        ab = tuple((str((a or {}).get("action_type") or "").lower(),
                    _coarse_bucket((a or {}).get("normalized_amount_bb")))
                   for a in acts)
        cbsig.append(cb)
        abcsig.append(ab)
    n = float(len(chunk))
    for tag, sig in (("amtbucketC", cbsig), ("actbucketC", abcsig)):
        out[f"sig_{tag}_top_share"] = _div(max(Counter(sig).values()), n)
        out[f"sig_{tag}_unique_share"] = _div(len(set(sig)), n)
    return out
