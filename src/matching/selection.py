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


def select(pairs, threshold, rel=0.0, max_matches=10):
    """pairs: DataFrame source1_entity_id, candidate_entity_id, p. Returns kept rows."""
    best = pairs.groupby("source1_entity_id")["p"].transform("max")
    kept = pairs[(pairs["p"] >= threshold) & (pairs["p"] >= rel * best)]
    kept = kept.sort_values(["source1_entity_id", "p"], ascending=[True, False])
    return kept[kept.groupby("source1_entity_id").cumcount() < max_matches]


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


def tune(pairs, entities, thresholds=None, rels=(0.0, 0.3, 0.5, 0.7), max_ms=(3, 5, 10)):
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
