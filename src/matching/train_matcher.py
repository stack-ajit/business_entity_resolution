"""
Train the two-stage matcher and tune the per-entity selection rule for macro F0.5.

Stage 1 (pruner): LightGBM on CHEAP_FEATURES only (retrieval + reverse context, no string
  work). Its threshold is set so that a target share of true pairs survives; survivors are
  the final candidate set (candidate_pairs.tsv). A smaller candidate set per S1 entity is
  explicitly rewarded in the final ranking, and it makes stage 2 several times cheaper.
Stage 2 (matcher): LightGBM on all FEATURES, trained on stage-1 survivors only - the same
  distribution it sees at test time.
Selection: threshold / relative-to-best / one-to-one resolution, tuned on the official
  macro F0.5 over ALL validation entities (incl. entities with no candidates).

Split by Source 1 entity (80/20), never by row. Final models are refit on all entities.

Usage: python src/matching/train_matcher.py [--stage1-recall 0.998]
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CACHE_DIR
from matching.pair_features import CHEAP_FEATURES, FEATURES
from matching.selection import macro_f05, select, tune

MODEL_DIR = os.path.join(CACHE_DIR, "model_v2")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


BASE = dict(objective="binary", feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=1.0, metric="binary_logloss", verbose=-1, num_threads=os.cpu_count())
P1 = dict(BASE, learning_rate=0.1, num_leaves=63, min_data_in_leaf=200)
P2 = dict(BASE, learning_rate=0.08, num_leaves=127, min_data_in_leaf=100)


def fit(params, tr, va, feats):
    m = lgb.train(params, lgb.Dataset(tr[feats], tr["label"]), num_boost_round=3000,
                  valid_sets=[lgb.Dataset(va[feats], va["label"])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    return m


def cands_per_entity(df, n_entities):
    return len(df) / n_entities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(CACHE_DIR, "train_pairs_v2.parquet"))
    ap.add_argument("--stage1-recall", type=float, default=0.998,
                    help="share of true candidate pairs stage 1 must keep")
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    pairs = pd.read_parquet(args.pairs)
    ents = pd.read_parquet(args.pairs.replace(".parquet", "_entities.parquet"))
    log(f"{len(pairs)} pairs ({cands_per_entity(pairs, len(ents)):.1f} per S1), {len(ents)} entities, "
        f"positive rate {pairs['label'].mean():.4f}")

    rng = np.random.default_rng(0)
    val_ids = set(ents["entity_id"].values[rng.random(len(ents)) < 0.2])
    is_val = pairs["source1_entity_id"].isin(val_ids).values
    tr, va = pairs[~is_val], pairs[is_val].copy()
    ents_va = ents[ents["entity_id"].isin(val_ids)]
    n_va = len(ents_va)

    # ---------------- stage 1: cheap pruner ----------------
    m1 = fit(P1, tr, va, CHEAP_FEATURES)
    va["p1"] = m1.predict(va[CHEAP_FEATURES], num_iteration=m1.best_iteration)
    pos_p1 = np.sort(va.loc[va["label"] == 1, "p1"].to_numpy())
    print("stage-1 operating points (validation):")
    for r in (0.99, 0.995, 0.998, 0.999):
        t = pos_p1[int((1 - r) * len(pos_p1))]
        kept = va[va["p1"] >= t]
        print(f"  keep {r:.3f} of true pairs -> threshold {t:.5f}, "
              f"{cands_per_entity(kept, n_va):.2f} candidates per S1")
    t1 = float(pos_p1[int((1 - args.stage1_recall) * len(pos_p1))])
    log(f"stage 1: best iteration {m1.best_iteration}, threshold {t1:.5f}")

    tr_p1 = m1.predict(tr[CHEAP_FEATURES], num_iteration=m1.best_iteration)
    tr2, va2 = tr[tr_p1 >= t1], va[va["p1"] >= t1].copy()
    log(f"stage-1 survivors: train {len(tr2)}, val {len(va2)} "
        f"({cands_per_entity(va2, n_va):.2f} per S1, was {cands_per_entity(va, n_va):.1f})")

    # ---------------- stage 2: full matcher ----------------
    m2 = fit(P2, tr2, va2, FEATURES)
    va2["p"] = m2.predict(va2[FEATURES], num_iteration=m2.best_iteration)
    log(f"stage 2: best iteration {m2.best_iteration}")

    score, params, table = tune(va2, ents_va)
    log(f"validation macro F0.5 = {score:.4f} with {params} (one-to-one resolution on)")
    print(table.sort_values("f05", ascending=False).head(8).to_string(index=False))
    no121 = macro_f05(select(va2, params["threshold"], params["rel"], params["max_matches"], one_to_one=False), ents_va)
    log(f"same rule without one-to-one resolution = {no121:.4f}")
    log(f"oracle F0.5, all candidates = {macro_f05(va[va['label'] == 1], ents_va):.4f}; "
        f"after stage 1 = {macro_f05(va2[va2['label'] == 1], ents_va):.4f}")

    imp = pd.Series(m2.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("stage-2 feature importance (gain, top 25):\n" + (imp / imp.sum()).round(4).head(25).to_string())

    # ---------------- refit on all entities ----------------
    log("refitting both stages on all entities")
    f1 = lgb.train(P1, lgb.Dataset(pairs[CHEAP_FEATURES], pairs["label"]),
                   num_boost_round=int(m1.best_iteration * 1.1))
    keep = f1.predict(pairs[CHEAP_FEATURES]) >= t1
    f2 = lgb.train(P2, lgb.Dataset(pairs.loc[keep, FEATURES], pairs.loc[keep, "label"]),
                   num_boost_round=int(m2.best_iteration * 1.1))
    f1.save_model(os.path.join(MODEL_DIR, "stage1.txt"))
    f2.save_model(os.path.join(MODEL_DIR, "stage2.txt"))
    with open(os.path.join(MODEL_DIR, "selection.json"), "w") as f:
        json.dump({**params, "stage1_threshold": t1, "stage1_recall": args.stage1_recall,
                   "val_macro_f05": score, "val_without_one_to_one": no121,
                   "val_cands_per_s1": cands_per_entity(va2, n_va),
                   "stage1_best_iteration": m1.best_iteration,
                   "stage2_best_iteration": m2.best_iteration}, f, indent=2)
    log("saved stage1/stage2 models + selection params")


if __name__ == "__main__":
    main()
