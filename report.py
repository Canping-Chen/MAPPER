from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy import stats

from .depth_baseline import auroc, depth_report, enrichment

def minimum_called_fraction(case: str, obs) -> float:
    total = 0.0
    for c in CRITERIA.get(case, ()):
        if c.kind != "recall" or c.threshold is None:
            continue
        share = float(np.mean(np.isin(np.asarray(obs[c.column]), c.population)))
        total += float(c.threshold) * share
    return total

SPEARMAN_PER_UNIT = 0.4
SPEARMAN_CELLTYPE_MEDIAN = 0.6

@dataclass(frozen=True)
class Criterion:
    name: str
    column: str
    population: tuple[str, ...]
    kind: str
    target: str
    threshold: float | None

    opposite_max: float | None = None

CRITERIA: dict[str, tuple[Criterion, ...]] = {}

_PRAD = (
    Criterion("NE_recall", "celltype", ("NE_cell",), "recall", "positive", 0.90),
    Criterion("mCAF_recall", "celltype", ("mCAFs",), "recall", "positive", 0.40),
    Criterion("SPP1_recall", "celltype", ("Mph.SPP1",), "recall", "positive", 0.40),
)
CRITERIA["prad_tcga"] = _PRAD
CRITERIA["prad_su2c"] = _PRAD
CRITERIA["hcc"] = (
    Criterion("tumour_precision", "Location", ("Mal", "Bdy"), "precision",
              "positive", 0.50),
)

for _case in ("gbm_03422", "gbm_25526"):
    CRITERIA[_case] = (
        Criterion("necrosis_recall", "Morphology", ("Necrotic", "Perinecrotic"),
                  "recall", "positive", None),
        Criterion("necrosis_precision", "Morphology", ("Necrotic", "Perinecrotic"),
                  "precision", "positive", None),
    )

def score_criterion(obs, group, score, criterion: Criterion, library,
                    n_detected=None, n_strata: int = 10, non_target=None) -> dict:
    group = np.asarray(group)
    score = np.asarray(score, dtype=float)
    if criterion.column not in obs:
        raise KeyError(
            f"criterion {criterion.name!r} needs obs column {criterion.column!r}; "
            f"available columns are {list(obs.columns)}"
        )
    member = np.isin(np.asarray(obs[criterion.column]), criterion.population)
    called = group == criterion.target

    opposite = {"positive": "negative", "negative": "positive"}.get(criterion.target)
    called_opposite = group == opposite if opposite else np.zeros(len(group), bool)
    opposite_observed = (float(np.mean(called_opposite[member]))
                         if member.any() else float("nan"))

    if criterion.kind == "recall":
        observed = float(np.mean(called[member])) if member.any() else float("nan")
        base = float(np.mean(called))
    elif criterion.kind == "precision":
        observed = float(np.mean(member[called])) if called.any() else float("nan")
        base = float(np.mean(member))
    else:
        raise ValueError(f"unknown criterion kind {criterion.kind!r}")

    row = {
        "name": criterion.name,
        "kind": criterion.kind,
        "population": list(criterion.population),
        "n": int(member.sum()),
        "threshold": criterion.threshold,
        "observed": observed,
        "base_rate": base,
        "enrichment": enrichment(observed, base),
        "called_fraction": float(np.mean(called)),
        "opposite_observed": opposite_observed,
        "opposite_max": criterion.opposite_max,
        "called_fraction_opposite": float(np.mean(called_opposite)),
    }

    if non_target is None:
        row["auroc_vs_non_target"] = float("nan")
    else:
        ref = np.asarray(non_target, dtype=bool)
        sel = member | ref
        row["auroc_vs_non_target"] = (
            auroc(score[sel], member[sel]) if member.any() and ref.any()
            else float("nan"))
        row["n_non_target"] = int(ref.sum())

    depth = depth_report(score, member, library, n_detected, n_strata=n_strata)

    depth["population_fraction"] = depth.pop("base_rate")

    depth["depth_control_usable"] = bool(depth["n_strata_available"] > 1)
    if not depth["depth_control_usable"]:
        depth["stratified_auroc"] = float("nan")
        depth["stratified_auroc_distance"] = float("nan")
    collisions = sorted(set(depth) & set(row))
    if collisions:
        raise RuntimeError(
            f"depth diagnostics would overwrite criterion fields {collisions}; "
            "rename one side rather than merging blindly"
        )
    row.update(depth)
    if criterion.threshold is None:
        row["passed"] = None
    else:
        ok = bool(np.isfinite(observed) and observed >= criterion.threshold)
        if criterion.opposite_max is not None:
            ok = ok and bool(np.isfinite(opposite_observed)
                             and opposite_observed <= criterion.opposite_max)
        row["passed"] = ok

    row["small_population"] = bool(row["n"] < 200)

    row["min_called_fraction_needed"] = (
        float(criterion.threshold) * float(np.mean(member))
        if criterion.kind == "recall" and criterion.threshold is not None
        else 0.0)
    row["operating_point_feasible"] = bool(
        row["called_fraction"] >= row["min_called_fraction_needed"] - 1e-12)
    return row

CRITERIA["atera_cesc"] = (
    Criterion("hypoxic_positive", "celltype", ("Hypoxic Tumor Cells",),
              "recall", "positive", 0.50, opposite_max=0.10),
    Criterion("cytotoxic_t_negative", "celltype", ("Cytotoxic T Cells",),
              "recall", "negative", 0.30, opposite_max=0.10),
)
CRITERIA["openst_hnsc"] = (
    Criterion("tumour_positive", "annotation", ("Tumor",),
              "recall", "positive", 0.50, opposite_max=0.10),
    Criterion("t_cell_negative", "annotation", ("T_cell",),
              "recall", "negative", 0.40, opposite_max=0.10),
)

OPERATING_POINTS = (0.05, 0.10, 0.20, 0.30, 0.40)

def criterion_at_operating_points(obs, score, criterion: Criterion,
                                  points=OPERATING_POINTS) -> dict:
    score = np.asarray(score, dtype=float).ravel()
    member = np.isin(np.asarray(obs[criterion.column]), criterion.population)
    finite = np.isfinite(score)
    out = {}
    for f in points:
        if not finite.any():
            out[f] = {"observed": float("nan"), "enrichment": float("nan")}
            continue
        usable = score[finite]
        if criterion.target == "negative":
            called = np.zeros(score.size, bool)
            called[np.flatnonzero(finite)] = usable <= np.quantile(usable, f)
        else:
            called = np.zeros(score.size, bool)
            called[np.flatnonzero(finite)] = usable >= np.quantile(usable, 1.0 - f)
        if criterion.kind == "recall":
            observed = float(np.mean(called[member])) if member.any() else float("nan")
            base = float(np.mean(called))
        else:
            observed = float(np.mean(member[called])) if called.any() else float("nan")
            base = float(np.mean(member))
        out[f] = {"observed": observed, "base_rate": base,
                  "enrichment": enrichment(observed, base)}
    return out

def score_case(case: str, obs, group, score, library, n_detected=None,
               n_strata: int = 10) -> list[dict]:
    if case not in CRITERIA:
        raise KeyError(f"unknown case {case!r}; declared cases are {sorted(CRITERIA)}")
    claimed = np.zeros(len(np.asarray(score)), dtype=bool)
    for c in CRITERIA[case]:
        if c.column in obs:
            claimed |= np.isin(np.asarray(obs[c.column]), c.population)
    non_target = ~claimed
    rows = []
    for c in CRITERIA[case]:
        row = score_criterion(obs, group, score, c, library, n_detected, n_strata,
                              non_target=non_target)
        row["operating_points"] = criterion_at_operating_points(obs, score, c)
        rows.append(row)
    return rows

def teacher_consistency(score_a, score_b, celltype) -> dict:
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"scores have different lengths: {a.shape} vs {b.shape}")
    per_unit = float(stats.spearmanr(a, b).statistic)

    celltype = np.asarray(celltype)
    labels = np.unique(celltype)
    med_a = np.array([np.median(a[celltype == c]) for c in labels])
    med_b = np.array([np.median(b[celltype == c]) for c in labels])
    per_type = float(stats.spearmanr(med_a, med_b).statistic) if labels.size > 2 else float("nan")

    return {
        "spearman_per_unit": per_unit,
        "spearman_celltype_median": per_type,
        "threshold_per_unit": SPEARMAN_PER_UNIT,
        "threshold_celltype_median": SPEARMAN_CELLTYPE_MEDIAN,
        "n_celltypes": int(labels.size),
        "passed": bool(per_unit > SPEARMAN_PER_UNIT
                       and per_type > SPEARMAN_CELLTYPE_MEDIAN),
    }

def isolated_positive_rate(group, adj) -> float:
    if adj is None:
        return float("nan")
    pos = np.asarray(group) == "positive"
    if not pos.any():
        return float("nan")
    off = sparse.csr_matrix(adj).copy()
    off.setdiag(0)
    off.eliminate_zeros()
    neighbours_positive = np.asarray(off @ pos.astype(float)).ravel()
    return float((neighbours_positive[pos] == 0).mean())

def target_contrast(obs, score, criteria, f: float) -> dict:
    score = np.asarray(score, dtype=float).ravel()
    finite = np.isfinite(score)
    idx = np.flatnonzero(finite)
    named = np.zeros(score.size, dtype=bool)
    sides = {"positive": np.zeros(score.size, dtype=bool),
             "negative": np.zeros(score.size, dtype=bool)}
    for c in criteria:
        member = np.isin(np.asarray(obs[c.column]), c.population)
        named |= member
        sides[c.target if c.target in sides else "positive"] |= member
    rest = ~named

    def _called(top: bool) -> np.ndarray:
        out = np.zeros(score.size, dtype=bool)
        if idx.size:
            usable = score[idx]
            out[idx] = (usable >= np.quantile(usable, 1.0 - f) if top
                        else usable <= np.quantile(usable, f))
        return out

    parts, detail = [], {}
    for side, top in (("positive", True), ("negative", False)):
        member = sides[side]
        if not member.any():
            detail[f"{side}_gap"] = float("nan")
            continue
        called = _called(top)
        t = float(called[member].mean())
        r = float(called[rest].mean()) if rest.any() else float("nan")
        detail[f"{side}_gap"] = t - r
        detail[f"{side}_recall"] = t
        detail[f"{side}_rest_recall"] = r
        detail[f"n_{side}"] = int(member.sum())
        parts.append(t - r)

    gap = float(np.mean(parts)) if parts else float("nan")
    top_called = _called(True)
    detail.update({
        "f": float(f), "gap": gap,
        "target_recall": (float(top_called[named].mean()) if named.any()
                          else float("nan")),
        "rest_recall": float(top_called[rest].mean()) if rest.any() else float("nan"),
        "n_target": int(named.sum()), "n_rest": int(rest.sum()),
    })
    return detail

def control_verdict(rows, gap=float("nan"), shuffled_gaps=(), chance: float = 1.0) -> dict:
    by = {}
    for row in rows:
        sh = [float(x) for x in
              np.asarray(row.get("shuffled_enrichment_at_f", []), dtype=float).ravel()
              if np.isfinite(x)]
        obs = float(row.get("enrichment_at_f", np.nan))
        worst = max(sh) if sh else float("nan")
        by[row.get("name", row.get("criterion"))] = {
            "enrichment": obs, "shuffled_max": worst,
            "margin": (obs - worst) if sh else float("nan"),
            "beats_shuffles": bool(sh) and np.isfinite(obs) and obs > worst,
            "above_chance": bool(np.isfinite(obs) and obs > chance),
            "n_shuffles": len(sh),
        }
    gaps = [float(x) for x in np.asarray(shuffled_gaps, dtype=float).ravel()
            if np.isfinite(x)]
    worst_gap = max(gaps) if gaps else float("nan")
    ok = bool(gaps) and bool(np.isfinite(gap)) and gap > 0.0 and gap > worst_gap
    return {"ok": ok, "gap": float(gap), "shuffled_gap_max": worst_gap,
            "margin": (gap - worst_gap) if gaps else float("nan"),
            "n_shuffles": len(gaps), "by_criterion": by}
