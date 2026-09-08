from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

NEGATIVE_TOLERANCE = 1e-6

CLINICAL_CONVENTIONS: tuple[tuple[str, Mapping[str, str]], ...] = (
    ("tcga", {"sample": "submitter_id", "time": "survival_time",
              "event": "vital_status"}),
    ("su2c", {"sample": "sample", "time": "OS_MONTHS", "event": "OS_STATUS"}),
)

ENDPOINTS = ("survival", "classification", "continuous")

@dataclass(frozen=True)
class BulkCohort:

    expr: np.ndarray
    genes: np.ndarray
    samples: np.ndarray
    endpoint: str
    time: np.ndarray | None = None
    event: np.ndarray | None = None
    label: np.ndarray | None = None
    value: np.ndarray | None = None

    @property
    def n_events(self) -> int:
        if self.endpoint == "survival":
            return int(np.sum(self.event))
        if self.endpoint == "classification":
            return int(np.sum(self.label))
        return int(self.value.shape[0])

def _duplicates(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    dup: list[str] = []
    for name in names:
        if name in seen and name not in dup:
            dup.append(name)
        seen.add(name)
    return dup

def _read_expression(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header = pd.read_csv(path, header=None, nrows=1).iloc[0].tolist()
    samples = [str(s) for s in header[1:]]
    dup = _duplicates(samples)
    if dup:
        raise ValueError(
            f"expression file has duplicate sample ids {dup[:5]}; refusing to "
            f"aggregate them because which column is which sample is ambiguous"
        )

    frame = pd.read_csv(path, index_col=0)
    genes = [str(g) for g in frame.index]
    dup = _duplicates(genes)
    if dup:
        raise ValueError(
            f"expression file has duplicate gene symbols {dup[:5]}; refusing to "
            f"aggregate them silently -- de-duplicate upstream and say how"
        )
    if frame.shape[1] != len(samples):
        raise ValueError(
            f"header lists {len(samples)} samples but {frame.shape[1]} columns "
            f"were parsed"
        )

    log2 = frame.to_numpy(dtype=float)
    if not np.isfinite(log2).all():
        raise ValueError("expression matrix contains non-finite values")
    if log2.min() < -NEGATIVE_TOLERANCE:
        raise ValueError(
            f"expression minimum is {log2.min():.4g}: these are not "
            f"log2(TPM+1) values, and inverting them would invent negative mass"
        )
    linear = np.maximum(np.exp2(log2) - 1.0, 0.0)
    return linear.T, np.asarray(genes, dtype=object), np.asarray(samples, dtype=object)

def _resolve_survival_columns(clin, sample_col, time_col, event_col):
    attempts = []
    for name, spec in CLINICAL_CONVENTIONS:
        s = sample_col or spec["sample"]
        t = time_col or spec["time"]
        v = event_col or spec["event"]
        missing = [c for c in (s, t, v) if c not in clin.columns]
        if not missing:
            explicit = all(c is not None for c in (sample_col, time_col, event_col))
            return ("explicit" if explicit else name), s, t, v
        attempts.append(f"{name}={ (s, t, v) } missing {missing}")
    raise ValueError(
        "cannot identify the survival columns; tried " + "; ".join(attempts)
        + f". Available columns: {list(clin.columns)}"
    )

def _resolve_sample_column(clin, sample_col) -> tuple[str, str]:
    if sample_col is not None:
        if sample_col not in clin.columns:
            raise ValueError(
                f"sample_col={sample_col!r} is not in the clinical table; "
                f"available columns: {list(clin.columns)}"
            )
        return "explicit", sample_col
    for name, spec in CLINICAL_CONVENTIONS:
        if spec["sample"] in clin.columns:
            return name, spec["sample"]
    raise ValueError(
        "cannot identify the sample id column; tried "
        + ", ".join(spec["sample"] for _, spec in CLINICAL_CONVENTIONS)
        + f". Available columns: {list(clin.columns)}"
    )

def _align(clin: pd.DataFrame, sample_col: str, samples: np.ndarray) -> pd.DataFrame:
    keys = clin[sample_col].astype(str)
    dup = _duplicates(list(keys))
    if dup:
        raise ValueError(f"clinical table has duplicate sample ids {dup[:5]}")
    indexed = clin.set_index(keys.to_numpy())
    known = set(indexed.index)
    missing = [s for s in samples if s not in known]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(samples)} expression samples are absent "
            f"from the clinical table, e.g. {missing[:5]}"
        )
    return indexed.loc[list(samples)]

def _survival(aligned, time_col, event_col):
    time = pd.to_numeric(aligned[time_col], errors="coerce").to_numpy(dtype=float)
    event = pd.to_numeric(aligned[event_col], errors="coerce").to_numpy(dtype=float)
    keep = ~(np.isnan(time) | np.isnan(event))
    time, event = time[keep], event[keep]
    if not np.isfinite(time).all():
        raise ValueError(f"column {time_col!r} holds non-finite follow-up times")
    if np.any(time < 0):
        raise ValueError(f"column {time_col!r} holds negative follow-up times")
    bad = np.setdiff1d(np.unique(event), np.array([0.0, 1.0]))
    if bad.size:
        raise ValueError(
            f"column {event_col!r} holds {bad.tolist()[:5]}; the event "
            f"indicator must be 0/1 with 1 meaning the event occurred"
        )
    return keep, {"time": time, "event": event.astype(int)}

def _classification(aligned, label_col, positive_class):
    raw = aligned[label_col]
    keep = raw.notna().to_numpy()
    vals = raw[keep]
    hit = (vals == positive_class) | (vals.astype(str) == str(positive_class))
    label = hit.to_numpy().astype(int)
    if label.sum() == 0:
        raise ValueError(
            f"positive_class={positive_class!r} matches none of the "
            f"{label.size} labelled samples; observed values are "
            f"{sorted(map(str, set(vals)))[:8]}"
        )
    return keep, {"label": label}

def _continuous(aligned, value_col):
    value = pd.to_numeric(aligned[value_col], errors="coerce").to_numpy(dtype=float)
    keep = ~np.isnan(value)
    value = value[keep]
    if not np.isfinite(value).all():
        raise ValueError(f"column {value_col!r} holds non-finite values")
    return keep, {"value": value}

def load_bulk(
    expr_csv,
    clin_csv,
    endpoint: str,
    *,
    time_col: str | None = None,
    event_col: str | None = None,
    label_col: str | None = None,
    positive_class: Any = None,
    value_col: str | None = None,
    sample_col: str | None = None,
    log=None,
) -> BulkCohort:
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown endpoint {endpoint!r}; expected one of {ENDPOINTS}")
    if endpoint == "classification":
        if label_col is None:
            raise ValueError("endpoint='classification' requires label_col")
        if positive_class is None:
            raise ValueError("endpoint='classification' requires positive_class")
    if endpoint == "continuous" and value_col is None:
        raise ValueError("endpoint='continuous' requires value_col")

    linear, genes, samples = _read_expression(expr_csv)

    order = np.argsort(samples.astype(str), kind="stable")
    linear, samples = linear[order], samples[order]
    sums = linear.sum(axis=1)
    if np.any(sums <= 0):
        empty = samples[sums <= 0][:5].tolist()
        raise ValueError(f"samples with no expressed mass: {empty}")
    expr = linear / sums[:, None]

    clin = pd.read_csv(clin_csv)
    if endpoint == "survival":
        convention, sample_col, time_col, event_col = _resolve_survival_columns(
            clin, sample_col, time_col, event_col)
    else:
        convention, sample_col = _resolve_sample_column(clin, sample_col)
        for role, col in (("label_col", label_col), ("value_col", value_col)):
            if col is not None and col not in clin.columns:
                raise ValueError(
                    f"{role}={col!r} is not in the clinical table; available "
                    f"columns: {list(clin.columns)}"
                )

    aligned = _align(clin, sample_col, samples)
    if endpoint == "survival":
        keep, fields = _survival(aligned, time_col, event_col)
    elif endpoint == "classification":
        keep, fields = _classification(aligned, label_col, positive_class)
    else:
        keep, fields = _continuous(aligned, value_col)

    if not keep.any():
        raise ValueError(
            f"every sample was dropped for a missing {endpoint} endpoint; the "
            f"clinical table and the expression matrix do not overlap in any "
            f"usable row"
        )
    cohort = BulkCohort(expr=expr[keep], genes=genes, samples=samples[keep],
                        endpoint=endpoint, **fields)

    if log is not None:
        quantiles = np.quantile(sums, [0.0, 0.25, 0.5, 0.75, 1.0])
        log.record("linear_sum_quantiles",
                   dict(zip(("min", "q25", "median", "q75", "max"), quantiles)),
                   section="bulk")
        log.record("linear_sum_max_over_min", float(sums.max() / sums.min()),
                   section="bulk")
        log.record("renormalised_per_sample", True, section="bulk")
        log.record("clinical_convention", convention, section="bulk")
        log.record("sample_column", sample_col, section="bulk")
        if endpoint == "survival":
            log.record("time_column", time_col, section="bulk")
            log.record("event_column", event_col, section="bulk")
        elif endpoint == "classification":
            log.record("label_column", label_col, section="bulk")
            log.record("positive_class", str(positive_class), section="bulk")
        else:
            log.record("value_column", value_col, section="bulk")
        log.record("n_dropped_missing_endpoint", int((~keep).sum()), section="bulk")
        log.record("endpoint", endpoint, section="bulk")
        log.record("n_samples", int(cohort.expr.shape[0]), section="bulk")
        log.record("n_genes", int(genes.shape[0]), section="bulk")
        log.record("n_events", cohort.n_events, section="bulk")
    return cohort
