"""
Train the pair classifier and tune the per-entity selection rule for macro F0.5.

- Split by Source 1 entity (80/20), never by row: all candidates of one entity stay on
  one side, exactly like unseen test entities.
- LightGBM (MIT) binary classifier on pair features; early stopping on validation.
- Selection rule (threshold / relative-to-best / max matches) is tuned by directly
  maximising the official macro F0.5 on validation entities, including entities with
  no candidates and true singletons.
- Final model is refit on all entities with the best iteration count.

Usage: python src/matching/train_matcher.py
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CACHE_DIR
from matching.pair_features import FEATURES
from matching.selection import macro_f05, select, tune

MODEL_DIR = os.path.join(CACHE_DIR, "model")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              metric="binary_logloss", verbose=-1, num_threads=os.cpu_count())


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    pairs = pd.read_parquet(os.path.join(CACHE_DIR, "train_pairs.parquet"))
    ents = pd.read_parquet(os.path.join(CACHE_DIR, "train_pairs_entities.parquet"))
    log(f"{len(pairs)} pairs, {len(ents)} entities, positive rate {pairs['label'].mean():.4f}")

    rng = np.random.default_rng(0)
    val_ids = set(ents["entity_id"].values[rng.random(len(ents)) < 0.2])
    is_val = pairs["source1_entity_id"].isin(val_ids).values
    tr, va = pairs[~is_val], pairs[is_val]
    ents_va = ents[ents["entity_id"].isin(val_ids)]

    dtr = lgb.Dataset(tr[FEATURES], tr["label"], free_raw_data=True)
    dva = lgb.Dataset(va[FEATURES], va["label"], reference=dtr)
    model = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    log(f"best iteration {model.best_iteration}")

    va = va[["source1_entity_id", "candidate_entity_id", "label"]].copy()
    va["p"] = model.predict(pairs.loc[is_val, FEATURES], num_iteration=model.best_iteration)
    score, params, table = tune(va, ents_va)
    log(f"validation macro F0.5 = {score:.4f} with {params}")
    print(table.sort_values("f05", ascending=False).head(10).to_string(index=False))

    # reference points to understand where the score comes from
    ceiling = macro_f05(va[va["label"] == 1], ents_va)
    log(f"oracle F0.5 given blocking (perfect classifier) = {ceiling:.4f}")
    log(f"all-empty baseline = {macro_f05(va.iloc[:0], ents_va):.4f}")
    log(f"top-1 if p>=0.5 = {macro_f05(select(va, 0.5, 0, 1), ents_va):.4f}")

    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("feature importance (gain):\n" + (imp / imp.sum()).round(4).to_string())

    log("refitting on all entities")
    full = lgb.train(PARAMS, lgb.Dataset(pairs[FEATURES], pairs["label"]),
                     num_boost_round=int(model.best_iteration * 1.1))
    full.save_model(os.path.join(MODEL_DIR, "lgb_matcher.txt"))
    with open(os.path.join(MODEL_DIR, "selection.json"), "w") as f:
        json.dump({**params, "val_macro_f05": score, "oracle_f05": ceiling,
                   "best_iteration": model.best_iteration}, f, indent=2)
    log("saved model + selection params")


if __name__ == "__main__":
    main()
