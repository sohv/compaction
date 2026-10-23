# cluster bootstrap for differences in per-arm rates and holm correction across named comparisons.
import numpy as np


def cluster_bootstrap_diff(
    rows_a: list[dict], rows_b: list[dict], cluster_key: str, value_key: str, n_resamples: int = 10000, seed: int = 42
) -> dict:
    """Mean of a minus mean of b, resampling clusters with replacement, the same draw for both arms."""
    clusters = sorted({r[cluster_key] for r in rows_a} | {r[cluster_key] for r in rows_b})
    index = {c: i for i, c in enumerate(clusters)}

    def per_cluster(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        sums, counts = np.zeros(len(clusters)), np.zeros(len(clusters))
        for r in rows:
            sums[index[r[cluster_key]]] += r[value_key]
            counts[index[r[cluster_key]]] += 1
        return sums, counts

    sums_a, counts_a = per_cluster(rows_a)
    sums_b, counts_b = per_cluster(rows_b)
    draws = np.random.default_rng(seed).integers(0, len(clusters), size=(n_resamples, len(clusters)))
    with np.errstate(invalid="ignore"):
        diffs = sums_a[draws].sum(1) / counts_a[draws].sum(1) - sums_b[draws].sum(1) / counts_b[draws].sum(1)
    diffs = diffs[~np.isnan(diffs)]
    return {
        "diff": float(sums_a.sum() / counts_a.sum() - sums_b.sum() / counts_b.sum()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
        "n_resamples": len(diffs),
    }


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()


def cluster_bootstrap_rate(
    rows: list[dict], cluster_key: str, value_key: str, n_resamples: int = 10000, seed: int = 42
) -> dict:
    """One arm's mean with a cluster-bootstrap interval, for the non-overlapping-interval bars."""
    zeros = [{cluster_key: r[cluster_key], value_key: 0} for r in rows]
    result = cluster_bootstrap_diff(rows, zeros, cluster_key, value_key, n_resamples, seed)
    return {"rate": result["diff"], "ci_low": result["ci_low"], "ci_high": result["ci_high"]}


def paired_permutation_p(rows_a: list[dict], rows_b: list[dict], cluster_key: str, value_key: str) -> float:
    """Exact sign-flip test on per-cluster rate differences; with 10 clusters there are 1024 sign patterns."""
    clusters = sorted({r[cluster_key] for r in rows_a} & {r[cluster_key] for r in rows_b})

    def rate(rows, c):
        vals = [r[value_key] for r in rows if r[cluster_key] == c]
        return sum(vals) / len(vals)

    d = np.array([rate(rows_a, c) - rate(rows_b, c) for c in clusters])
    observed = abs(d.mean())
    signs = np.array(np.meshgrid(*[[-1, 1]] * len(d))).reshape(len(d), -1).T
    return float((np.abs((signs * d).mean(1)) >= observed - 1e-12).mean())
