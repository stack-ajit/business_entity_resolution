"""
Turn per-pair match probabilities into per-entity match lists, and score them with the
official metric (macro F0.5 over ALL Source 1 entities, singletons included).

Selection rule (tuned on validation, applied unchanged at test time):
    keep candidate  if  p >= threshold  and  p >= rel * (best p of that entity)
    and at most max_matches per entity (by p).
An entity with nothing kept gets an empty list, which is worth 1.0 if it truly is a singleton.
"""
import itertools

import numpy as np
import pandas as pd


def resolve_one_to_one(pairs):
    """Each S2/S3 record belongs to at most one S1 (true in the data): keep only its best-p claim."""
    if pairs.empty:
        return pairs
    return pairs.loc[pairs.groupby("candidate_entity_id")["p"].idxmax()]


def select(pairs, threshold, rel=0.0, max_matches=10, one_to_one=True):
    """pairs: DataFrame source1_entity_id, candidate_entity_id, p. Returns kept rows."""
    best = pairs.groupby("source1_entity_id")["p"].transform("max")
    kept = pairs[(pairs["p"] >= threshold) & (pairs["p"] >= rel * best)]
    if one_to_one:
        kept = resolve_one_to_one(kept)
    kept = kept.sort_values(["source1_entity_id", "p"], ascending=[True, False])
    return kept[kept.groupby("source1_entity_id").cumcount() < max_matches]


def select_expected(pairs, miss=0.1, power=1.0, floor=0.02, one_to_one=True):
    """
    Per-entity expected-F0.5-optimal selection. For each S1, candidates sorted by p; keeping
    the top k scores in expectation
        E[F0.5 | k] ~ 1.25 * sum(top-k p) / (k + 0.25 * N),  N = sum(all p) + miss
    and keeping nothing scores P(no true match) ~ prod(1 - p) (a correct singleton = 1.0).
    Pick the best k. Adapts per entity instead of one global threshold.
    miss: expected true matches outside the candidate set (blocking loss).
    power: calibration tweak, p -> p**power (tuned on validation).
    """
    q = pairs[pairs["p"] >= floor].copy()
    if one_to_one:
        q = resolve_one_to_one(q)
    q["q"] = q["p"].clip(1e-6, 1 - 1e-6) ** power
    q = q.sort_values(["source1_entity_id", "q"], ascending=[True, False])
    g = q.groupby("source1_entity_id")["q"]
    k = g.cumcount() + 1
    cum = g.cumsum()
    n_exp = g.transform("sum") + miss
    ef = 1.25 * cum / (k + 0.25 * n_exp)
    e0 = np.exp(np.log1p(-q["q"]).groupby(q["source1_entity_id"]).transform("sum"))
    best_ef = ef.groupby(q["source1_entity_id"]).transform("max")
    k_best = k.where(ef == best_ef).groupby(q["source1_entity_id"]).transform("min")
    keep = (k <= k_best) & (best_ef > e0)
    return q[keep].drop(columns="q")


def tune_expected(pairs, entities, misses=(0.0, 0.1, 0.2, 0.3, 0.5), powers=(0.8, 1.0, 1.2, 1.5)):
    rows = [(macro_f05(select_expected(pairs, m, pw), entities), m, pw)
            for m, pw in itertools.product(misses, powers)]
    table = pd.DataFrame(rows, columns=["f05", "miss", "power"])
    best = table.sort_values("f05", ascending=False).iloc[0]
    return float(best.f05), {"miss": float(best.miss), "power": float(best.power)}, table


def macro_f05(kept, entities):
    """
    kept: selected pairs with a 'label' column. entities: DataFrame entity_id, n_true_total
    covering EVERY evaluated S1 entity (including ones with no candidates).
    """
    g = kept.groupby("source1_entity_id")["label"].agg(["sum", "size"])
    e = entities.set_index("entity_id")
    tp = g["sum"].reindex(e.index, fill_value=0).to_numpy(float)
    n_pred = g["size"].reindex(e.index, fill_value=0).to_numpy(float)
    n_true = e["n_true_total"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, tp / n_pred, 0.0)
        r = np.where(n_true > 0, tp / n_true, 0.0)
        f = np.where((p + r) > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, f)  # correct singleton
    return float(f.mean())


def tune(pairs, entities, thresholds=None, rels=(0.0, 0.5, 0.7), max_ms=(10,)):
    """Grid-search the selection rule on validation; returns (best_score, params, table)."""
    thresholds = np.round(np.arange(0.2, 0.91, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t, r, m in itertools.product(thresholds, rels, max_ms):
        rows.append((macro_f05(select(pairs, t, r, m), entities), t, r, m))
    table = pd.DataFrame(rows, columns=["f05", "threshold", "rel", "max_matches"])
    best = table.sort_values("f05", ascending=False).iloc[0]
    params = {"threshold": float(best.threshold), "rel": float(best.rel),
              "max_matches": int(best.max_matches)}
    return float(best.f05), params, table
