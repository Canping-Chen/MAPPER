from __future__ import annotations

import numpy as np

POSITIVE, BACKGROUND, NEGATIVE = "positive", "background", "negative"

def _reference_level(score: np.ndarray, group: np.ndarray) -> float:
    score = np.asarray(score, dtype=float).ravel()
    group = np.asarray(group, dtype=bool)
    if not group.any():
        raise ValueError("reference group is empty")
    return float(np.median(score[group]))

def anchor_to_cohort(unit_score, niche_score, pseudobulk_score: float,
                     log=None):
    unit = np.asarray(unit_score, dtype=float).ravel()
    anchor = float(pseudobulk_score)
    if not np.isfinite(anchor):
        raise ValueError("the pooled-profile anchor is not finite; the slide "
                         "cannot be placed on the cohort's scale")
    finite = np.isfinite(unit)
    if not finite.any():
        raise ValueError("no finite unit scores to anchor")
    shift = float(unit[finite].mean() - anchor)
    out_n = None
    if niche_score is not None:
        out_n = np.asarray(niche_score, dtype=float).ravel() - shift
    if log is not None:
        log.record("anchor_pooled_profile", round(anchor, 6), section="calls")
        log.record("anchor_shift", round(shift, 6), section="calls")
    return unit - shift, out_n, shift

def reference_scores(teacher, w_bulk, clinical, endpoint: str,
                     frame: str = "offset", log=None):
    if frame not in ("offset", "cohort"):
        raise ValueError(f"unknown frame {frame!r}; expected 'offset' or 'cohort'")
    held = getattr(teacher, "detail", {}).get("oof")
    s = np.asarray(held, dtype=float) if held is not None else teacher.score(w_bulk)
    if endpoint == "survival":
        t = np.asarray(clinical["time"], float)
        e = np.asarray(clinical["event"], float)
        land = float(np.nanquantile(t, 0.5))
        bad, good = (e >= 0.5) & (t <= land), (e < 0.5) & (t > land)
    elif endpoint == "classification":
        y = np.asarray(clinical["label"]).astype(bool)
        bad, good = y, ~y
    else:
        v = np.asarray(clinical["value"], float)
        bad, good = v >= np.quantile(v, 0.75), v <= np.quantile(v, 0.25)
    if bad.sum() < 3 or good.sum() < 3:
        raise ValueError(
            f"the landmark leaves {int(bad.sum())} bad and {int(good.sum())} good "
            f"samples; the reference groups are too small to place a threshold"
        )
    seen = np.isfinite(s)
    bad, good = bad & seen, good & seen
    sign = 1.0
    if _reference_level(s, bad) < _reference_level(s, good):
        sign, s = -1.0, -s
    hi, lo = _reference_level(s, bad), _reference_level(s, good)
    centre = float(np.median(s[seen]))
    spread = float(np.std(s[seen])) or 1.0
    if frame == "offset":
        hi, lo = (hi - centre) / spread, (lo - centre) / spread
    if log is not None:
        log.record("reference_frame", frame, section="calls")
        log.record("cohort_typical_patient", round(centre, 4), section="calls")
        log.record("cohort_spread", round(spread, 4), section="calls")
        log.record("landmark_bad_n", int(bad.sum()), section="calls")
        log.record("landmark_good_n", int(good.sum()), section="calls")
        log.record("teacher_orientation", sign, section="calls")
        log.record("threshold_bad_group", round(hi, 4), section="calls")
        log.record("threshold_good_group", round(lo, 4), section="calls")
        log.record("threshold_rule", "median score of the reference group",
                   section="calls")
        log.record("threshold_read_on",
                   "out-of-fold cohort scores" if held is not None
                   else "in-sample cohort scores", section="calls")
        log.record("bad_above_threshold", round(float((s[bad] >= hi).mean()), 3),
                   section="calls")
        log.record("good_below_threshold", round(float((s[good] <= lo).mean()), 3),
                   section="calls")
    return hi, lo, sign

def call(unit_score, niche_score, hi: float, lo: float, rule: str = "mean",
         niche_weight: float = 0.5, log=None) -> np.ndarray:
    if rule not in ("either", "both", "mean"):
        raise ValueError(f"unknown rule {rule!r}; expected 'either', 'both' or 'mean'")
    unit = np.asarray(unit_score, float)
    niche = np.asarray(niche_score, float)
    group = np.full(unit.shape[0], BACKGROUND, dtype=object)

    u_ok, n_ok = np.isfinite(unit), np.isfinite(niche)
    if rule == "mean":
        w = float(niche_weight)
        joint = np.where(n_ok,
                         (1.0 - w) * unit + w * np.where(n_ok, niche, 0.0),
                         unit)
        high, low = u_ok & (joint >= hi), u_ok & (joint <= lo)
    elif rule == "both":
        high = (u_ok & (unit >= hi)) & np.where(n_ok, niche >= hi, True)
        low = (u_ok & (unit <= lo)) & np.where(n_ok, niche <= lo, True)
    else:
        high = (u_ok & (unit >= hi)) | (n_ok & (niche >= hi))
        low = (u_ok & (unit <= lo)) | (n_ok & (niche <= lo))
    group[low] = NEGATIVE
    group[high] = POSITIVE

    if log is not None:
        log.record("call_rule", rule, section="calls")
        if rule == "mean":
            log.record("call_niche_weight", float(niche_weight), section="calls")
        log.record("positive", float(np.mean(group == POSITIVE)), section="calls")
        log.record("negative", float(np.mean(group == NEGATIVE)), section="calls")
        log.record("background", float(np.mean(group == BACKGROUND)), section="calls")
        log.record("n_both_sides", int((high & low).sum()), section="calls")
        log.record("n_without_niche", int((~n_ok).sum()), section="calls")
        log.record("positive_from_unit_only",
                   int(((u_ok & (unit >= hi)) & ~(n_ok & (niche >= hi))).sum()),
                   section="calls")
        log.record("positive_from_niche_only",
                   int((~(u_ok & (unit >= hi)) & (n_ok & (niche >= hi))).sum()),
                   section="calls")
    return group
