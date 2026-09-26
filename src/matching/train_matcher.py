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
from matching.pair_features import CHEAP_FEATURES, FEATURES, S3_FEATURES, second_order_features
from matching.selection import macro_f05, select, select_expected, tune, tune_expected

MODEL_DIR = os.path.join(CACHE_DIR, "model_v4")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


BASE = dict(objective="binary", feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=1.0, metric="binary_logloss", verbose=-1, num_threads=os.cpu_count())
P1 = dict(BASE, learning_rate=0.1, num_leaves=63, min_data_in_leaf=200)
P2 = dict(BASE, learning_rate=0.08, num_leaves=127, min_data_in_leaf=100)
P3 = dict(BASE, learning_rate=0.05, num_leaves=63, min_data_in_leaf=100)


def oof(df, feats, n_iter, k=3, seed=0):
    """Out-of-fold stage-2 probabilities, folds split by S1 entity (no within-entity leakage)."""
    ents = df["source1_entity_id"].unique()
    fold_of = dict(zip(ents, np.random.default_rng(seed).integers(0, k, len(ents))))
    fold = df["source1_entity_id"].map(fold_of).to_numpy()
    out = np.zeros(len(df))
    for i in range(k):
        m = lgb.train(P2, lgb.Dataset(df.loc[fold != i, feats], df.loc[fold != i, "label"]),
                      num_boost_round=n_iter)
        out[fold == i] = m.predict(df.loc[fold == i, feats])
    return out


def fit(params, tr, va, feats):
    m = lgb.train(params, lgb.Dataset(tr[feats], tr["label"]), num_boost_round=3000,
                  valid_sets=[lgb.Dataset(va[feats], va["label"])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    return m


def cands_per_entity(df, n_entities):
    return len(df) / n_entities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(CACHE_DIR, "train_pairs_v4.parquet"))
    ap.add_argument("--stage1-recall", type=float, default=0.998,
                    help="share of true candidate pairs stage 1 must keep")
    ap.add_argument("--stage3", choices=["auto", "on", "off"], default="auto",
                    help="auto: use stage 3 only if it wins on validation")
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
    it2 = m2.best_iteration
    va2["p2"] = m2.predict(va2[FEATURES], num_iteration=it2)
    log(f"stage 2: best iteration {it2}")
    imp = pd.Series(m2.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("stage-2 feature importance (gain, top 25):")
    print((imp / imp.sum()).round(4).head(25).to_string())

    # ---------------- stage 3: second-order (entity context + sibling consistency) ----------------
    # trained on OUT-OF-FOLD stage-2 probabilities so it never sees leaked, over-confident p2
    edges = pd.read_parquet(args.pairs.replace(".parquet", "_edges.parquet"))
    tr2 = tr2.copy()
    tr2["p2"] = oof(tr2, FEATURES, it2)
    log("stage-2 out-of-fold probabilities done")
    tr3, va3 = second_order_features(tr2, edges), second_order_features(va2, edges)
    m3 = fit(P3, tr3, va3, FEATURES + S3_FEATURES)
    it3 = m3.best_iteration
    va3["p3"] = m3.predict(va3[FEATURES + S3_FEATURES], num_iteration=it3)
    log(f"stage 3: best iteration {it3}")
    imp3 = pd.Series(m3.feature_importance("gain"), index=FEATURES + S3_FEATURES).sort_values(ascending=False)
    print("stage-3 feature importance (gain, top 12):")
    print((imp3 / imp3.sum()).round(4).head(12).to_string())

    # ---------------- selection: pick the best (stage, rule) on validation ----------------
    results = {}
    for stage, col in (("stage2", "p2"), ("stage3", "p3")):
        vv = va3.assign(p=va3[col])
        t_score, t_params, _ = tune(vv, ents_va)
        e_score, e_params, _ = tune_expected(vv, ents_va)
        no121 = macro_f05(select(vv, t_params["threshold"], t_params["rel"], t_params["max_matches"],
                                 one_to_one=False), ents_va)
        log(f"{stage}: threshold rule {t_score:.4f} {t_params} | expected-F0.5 {e_score:.4f} {e_params} "
            f"| threshold rule w/o one-to-one {no121:.4f}")
        results[stage] = (max(t_score, e_score), {**t_params, **e_params,
                          "mode": "expected" if e_score > t_score else "threshold"})
    best_stage = max(results, key=lambda k: results[k][0])
    if args.stage3 != "auto":
        best_stage = "stage3" if args.stage3 == "on" else "stage2"
    score, params = results[best_stage]
    log(f"validation macro F0.5 = {score:.4f} using {best_stage}, {params}")
    log(f"oracle F0.5, all candidates = {macro_f05(va[va['label'] == 1], ents_va):.4f}; "
        f"after stage 1 = {macro_f05(va2[va2['label'] == 1], ents_va):.4f}")

    # ---------------- refit on all entities ----------------
    log("refitting on all entities")
    f1 = lgb.train(P1, lgb.Dataset(pairs[CHEAP_FEATURES], pairs["label"]),
                   num_boost_round=int(m1.best_iteration * 1.1))
    keep = f1.predict(pairs[CHEAP_FEATURES]) >= t1
    surv = pairs[keep].copy()
    f2 = lgb.train(P2, lgb.Dataset(surv[FEATURES], surv["label"]), num_boost_round=int(it2 * 1.1))
    f1.save_model(os.path.join(MODEL_DIR, "stage1.txt"))
    f2.save_model(os.path.join(MODEL_DIR, "stage2.txt"))
    if best_stage == "stage3":
        surv["p2"] = oof(surv, FEATURES, int(it2 * 1.1))
        s3 = second_order_features(surv, edges)
        f3 = lgb.train(P3, lgb.Dataset(s3[FEATURES + S3_FEATURES], s3["label"]),
                       num_boost_round=int(it3 * 1.1))
        f3.save_model(os.path.join(MODEL_DIR, "stage3.txt"))
    with open(os.path.join(MODEL_DIR, "selection.json"), "w") as f:
        json.dump({**params, "use_stage3": best_stage == "stage3",
                   "stage1_threshold": t1, "stage1_recall": args.stage1_recall,
                   "val_macro_f05": score, "val_by_stage": {k: v[0] for k, v in results.items()},
                   "val_cands_per_s1": cands_per_entity(va2, n_va),
                   "stage1_best_iteration": m1.best_iteration, "stage2_best_iteration": it2,
                   "stage3_best_iteration": it3}, f, indent=2)
    log(f"saved models ({'stage1-3' if best_stage == 'stage3' else 'stage1-2'}) + selection params")


if __name__ == "__main__":
    main()
