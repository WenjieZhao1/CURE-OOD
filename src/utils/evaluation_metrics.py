from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _finite_1d(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _paired_finite(labels: Sequence[int], scores: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels).reshape(-1)
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    if labels_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError("labels and scores must have the same length")
    mask = np.isfinite(scores_arr)
    return labels_arr[mask].astype(int), scores_arr[mask]


def _nan_binary_metrics() -> Dict[str, float]:
    return {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")}


def _ci_bounds(values: Sequence[float], ci_level: float) -> Tuple[float, float]:
    arr = _finite_1d(values)
    if arr.size == 0:
        return float("nan"), float("nan")
    alpha = max(0.0, min(1.0, 1.0 - float(ci_level)))
    low_pct = 100.0 * alpha / 2.0
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(arr, low_pct)), float(np.percentile(arr, high_pct))


def _std(values: Sequence[float]) -> float:
    arr = _finite_1d(values)
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1))


def _empty_bootstrap_result(
    point: Mapping[str, float],
    n_bootstrap: int,
    n_id: int,
    n_ood: int,
    positive_label: str,
) -> Dict[str, float]:
    result = {
        "auroc": float(point.get("auroc", float("nan"))),
        "aupr": float(point.get("aupr", float("nan"))),
        "fpr95": float(point.get("fpr95", float("nan"))),
        "auroc_ci_low": float("nan"),
        "auroc_ci_high": float("nan"),
        "auroc_boot_std": float("nan"),
        "aupr_ci_low": float("nan"),
        "aupr_ci_high": float("nan"),
        "aupr_boot_std": float("nan"),
        "fpr95_ci_low": float("nan"),
        "fpr95_ci_high": float("nan"),
        "fpr95_boot_std": float("nan"),
        "n_bootstrap": int(n_bootstrap),
        "n_bootstrap_valid": 0,
        "n_id": int(n_id),
        "n_ood": int(n_ood),
        "positive_label": positive_label,
        "score_orientation": "higher_score_is_positive_label",
    }
    return result


def compute_binary_ood_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    labels_arr, scores_arr = _paired_finite(labels, scores)
    if labels_arr.size == 0 or np.unique(labels_arr).size < 2:
        return _nan_binary_metrics()

    metrics = _nan_binary_metrics()
    try:
        metrics["auroc"] = float(roc_auc_score(labels_arr, scores_arr))
    except ValueError:
        pass
    try:
        metrics["aupr"] = float(average_precision_score(labels_arr, scores_arr))
    except ValueError:
        pass
    try:
        fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
        metrics["fpr95"] = float(np.interp(0.95, tpr, fpr)) if np.any(tpr >= 0.95) else 1.0
    except ValueError:
        pass
    return metrics


def bootstrap_ood_metrics(
    id_scores: Sequence[float],
    ood_scores: Sequence[float],
    n_bootstrap: int,
    ci_level: float = 0.95,
    seed: int = 786,
    ood_positive: bool = True,
) -> Dict[str, float]:
    id_arr = _finite_1d(id_scores)
    ood_arr = _finite_1d(ood_scores)
    positive_label = "OOD" if ood_positive else "ID"

    id_label = 0 if ood_positive else 1
    ood_label = 1 if ood_positive else 0
    labels = np.concatenate(
        [
            np.full(id_arr.shape[0], id_label, dtype=int),
            np.full(ood_arr.shape[0], ood_label, dtype=int),
        ]
    )
    scores = np.concatenate([id_arr, ood_arr])
    point = compute_binary_ood_metrics(labels, scores)

    if n_bootstrap <= 0 or id_arr.size == 0 or ood_arr.size == 0:
        return _empty_bootstrap_result(point, n_bootstrap, id_arr.size, ood_arr.size, positive_label)

    rng = np.random.default_rng(seed)
    boot_values = {"auroc": [], "aupr": [], "fpr95": []}

    for _ in range(int(n_bootstrap)):
        id_sample = id_arr[rng.integers(0, id_arr.size, size=id_arr.size)]
        ood_sample = ood_arr[rng.integers(0, ood_arr.size, size=ood_arr.size)]
        sample_labels = np.concatenate(
            [
                np.full(id_sample.shape[0], id_label, dtype=int),
                np.full(ood_sample.shape[0], ood_label, dtype=int),
            ]
        )
        sample_scores = np.concatenate([id_sample, ood_sample])
        sample_metrics = compute_binary_ood_metrics(sample_labels, sample_scores)
        if all(math.isfinite(sample_metrics[key]) for key in boot_values):
            for key in boot_values:
                boot_values[key].append(sample_metrics[key])

    valid = len(boot_values["auroc"])
    result = _empty_bootstrap_result(point, n_bootstrap, id_arr.size, ood_arr.size, positive_label)
    result["n_bootstrap_valid"] = int(valid)

    for metric_name in ("auroc", "aupr", "fpr95"):
        low, high = _ci_bounds(boot_values[metric_name], ci_level)
        result[f"{metric_name}_ci_low"] = low
        result[f"{metric_name}_ci_high"] = high
        result[f"{metric_name}_boot_std"] = _std(boot_values[metric_name])

    return result


def censoring_summary(events: Sequence[float]) -> Dict[str, float]:
    event_arr = np.asarray(events, dtype=float).reshape(-1)
    event_arr = event_arr[np.isfinite(event_arr)]
    n = int(event_arr.size)
    n_events = int(np.sum(event_arr > 0)) if n else 0
    n_censored = int(n - n_events)
    event_rate = float(n_events / n) if n else float("nan")
    censoring_rate = float(n_censored / n) if n else float("nan")
    return {
        "n": n,
        "n_events": n_events,
        "n_censored": n_censored,
        "event_rate": event_rate,
        "censoring_rate": censoring_rate,
    }


def bootstrap_cindex(
    times: Sequence[float],
    risks: Sequence[float],
    events: Sequence[float],
    n_bootstrap: int,
    ci_level: float = 0.95,
    seed: int = 786,
) -> Dict[str, float]:
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    risks_arr = np.asarray(risks, dtype=float).reshape(-1)
    events_arr = np.asarray(events, dtype=float).reshape(-1)
    if not (times_arr.shape[0] == risks_arr.shape[0] == events_arr.shape[0]):
        raise ValueError("times, risks, and events must have the same length")

    valid_mask = np.isfinite(times_arr) & np.isfinite(risks_arr) & np.isfinite(events_arr)
    times_arr = times_arr[valid_mask]
    risks_arr = risks_arr[valid_mask]
    events_arr = events_arr[valid_mask]
    summary = censoring_summary(events_arr)

    try:
        point = float(concordance_index(times_arr, -risks_arr, event_observed=events_arr))
    except Exception:
        point = float("nan")

    result = {
        "cindex": point,
        "cindex_ci_low": float("nan"),
        "cindex_ci_high": float("nan"),
        "cindex_boot_std": float("nan"),
        "n_bootstrap": int(n_bootstrap),
        "n_bootstrap_valid": 0,
        **summary,
    }

    if n_bootstrap <= 0 or times_arr.size < 2:
        return result

    rng = np.random.default_rng(seed)
    boot = []
    n = times_arr.size
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        try:
            val = float(concordance_index(times_arr[idx], -risks_arr[idx], event_observed=events_arr[idx]))
        except Exception:
            continue
        if math.isfinite(val):
            boot.append(val)

    low, high = _ci_bounds(boot, ci_level)
    result["cindex_ci_low"] = low
    result["cindex_ci_high"] = high
    result["cindex_boot_std"] = _std(boot)
    result["n_bootstrap_valid"] = int(len(boot))
    return result


def extended_score_family_from_logits(
    task_logits: Mapping[str, Sequence[float]],
) -> List[Tuple[str, np.ndarray]]:
    entries: List[Tuple[str, np.ndarray]] = []
    for task_name, logits in task_logits.items():
        arr = np.asarray(logits, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"logits for {task_name} must be 2D")
        shifted = arr - np.max(arr, axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        max_prob = np.max(probs, axis=1)
        entries.append((f"MaxEventProb_{task_name}", max_prob))
        entries.append((f"OneMinusMaxEventProb_{task_name}", 1.0 - max_prob))
    return entries


def hazard_l1_l2_entries(
    task_hazard_diff: Mapping[str, Sequence[float]],
) -> List[Tuple[str, np.ndarray]]:
    entries: List[Tuple[str, np.ndarray]] = []
    for task_name, diff in task_hazard_diff.items():
        arr = np.asarray(diff, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"hazard diff for {task_name} must be 2D")
        entries.append((f"HazardDevL1_{task_name}", np.mean(np.abs(arr), axis=1)))
        entries.append((f"HazardDevL2_{task_name}", np.sqrt(np.mean(np.square(arr), axis=1))))
    return entries


def mixed_ood_metric_row(
    score_name: str,
    values: Sequence[float],
    binary_targets: Sequence[int],
    id_mask: Sequence[bool],
    ood_mask: Sequence[bool],
    n_bootstrap: int = 0,
    ci_level: float = 0.95,
    seed: int = 786,
) -> Dict[str, float]:
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    targets_arr = np.asarray(binary_targets, dtype=int).reshape(-1)
    id_mask_arr = np.asarray(id_mask, dtype=bool).reshape(-1)
    ood_mask_arr = np.asarray(ood_mask, dtype=bool).reshape(-1)

    row = {
        "score": score_name,
        "mean_id": float(np.nanmean(values_arr[id_mask_arr])) if id_mask_arr.any() else float("nan"),
        "mean_ood": float(np.nanmean(values_arr[ood_mask_arr])) if ood_mask_arr.any() else float("nan"),
    }

    metrics = compute_binary_ood_metrics(targets_arr, values_arr)
    row.update(metrics)

    if n_bootstrap > 0:
        boot = bootstrap_ood_metrics(
            values_arr[id_mask_arr],
            values_arr[ood_mask_arr],
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
            ood_positive=True,
        )
        for key in (
            "auroc_ci_low",
            "auroc_ci_high",
            "auroc_boot_std",
            "aupr_ci_low",
            "aupr_ci_high",
            "aupr_boot_std",
            "fpr95_ci_low",
            "fpr95_ci_high",
            "fpr95_boot_std",
            "n_bootstrap",
            "n_bootstrap_valid",
            "n_id",
            "n_ood",
        ):
            row[key] = boot[key]

    return row


def cindex_metric_rows(
    task_specs: Iterable[Tuple[str, Sequence[float], Sequence[float], Sequence[float]]],
    domain_labels: Sequence[int],
    id_label: int,
    ood_label: int,
    n_bootstrap: int,
    ci_level: float = 0.95,
    seed: int = 786,
) -> List[Dict[str, float]]:
    domain_arr = np.asarray(domain_labels).reshape(-1)
    split_masks = [
        ("ALL", np.ones(domain_arr.shape[0], dtype=bool)),
        ("ID", domain_arr == id_label),
        ("OOD", domain_arr == ood_label),
    ]

    rows: List[Dict[str, float]] = []
    for task_idx, (task_name, times, risks, events) in enumerate(task_specs):
        times_arr = np.asarray(times, dtype=float).reshape(-1)
        risks_arr = np.asarray(risks, dtype=float).reshape(-1)
        events_arr = np.asarray(events, dtype=float).reshape(-1)
        for split_idx, (split_name, mask) in enumerate(split_masks):
            metrics = bootstrap_cindex(
                times_arr[mask],
                risks_arr[mask],
                events_arr[mask],
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=seed + task_idx * 1009 + split_idx,
            )
            rows.append(
                {
                    "task": task_name,
                    "split": split_name,
                    **metrics,
                    "cindex_type": "lifelines.utils.concordance_index",
                    "risk_definition": "torchmtlr.mtlr_risk",
                    "score_passed_to_lifelines": "-risk",
                    "ipcw": False,
                }
            )
    return rows
