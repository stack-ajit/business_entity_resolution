"""
Build the labelled pair dataset for the matching model.

Takes a random subset of train Source 1 entities, runs the SAME blocking used at test
time against the full train Source 2/3 index, labels each candidate with the ground
truth, and computes pair features. Training on real blocking output (not random
negatives) means the model learns to reject exactly the look-alikes it will face.

Usage:
    python src/matching/build_training_set.py --n-s1 150000 --top-k 50
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from blocking.tfidf_blocking import query_index, read_tsv_chunks
from matching.pair_features import load_records, pair_features

from config import TRAIN_DIR as TRAIN, CACHE_DIR as CACHE, TRAIN_INDEX
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=150_000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(CACHE, "train_pairs.parquet"))
    args = ap.parse_args()

    s1 = pd.concat(read_tsv_chunks(os.path.join(TRAIN, "train_source1.tsv")), ignore_index=True)
    s1 = s1.sample(n=args.n_s1, random_state=args.seed).reset_index(drop=True)
    log(f"sampled {len(s1)} S1 entities")

    cands = query_index(s1["entity_id"].values, s1["business_name"].values,
                        s1["business_address"].values, s1["country"].values,
                        TRAIN_INDEX, top_k=args.top_k, log=log)
    log(f"{len(cands)} candidate pairs")

    gt = pd.concat(read_tsv_chunks(os.path.join(TRAIN, "train_ground_truth.tsv")), ignore_index=True)
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
    truth = {(a, b) for a, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
             for b in m.split(",") if b}
    n_true = gt.set_index("source1_entity_id")["matched_entity_ids"].map(
        lambda m: len([x for x in m.split(",") if x]))

    s1_rec = load_records([os.path.join(TRAIN, "train_source1.tsv")], s1["entity_id"])
    s23_rec = load_records([os.path.join(TRAIN, f"train_source{i}.tsv") for i in (2, 3)],
                           cands["candidate_entity_id"].unique())
    log("records loaded, computing features")
    F = pair_features(cands, s1_rec, s23_rec)
    F["label"] = [int((a, b) in truth) for a, b in zip(F["source1_entity_id"], F["candidate_entity_id"])]
    F["n_true_total"] = F["source1_entity_id"].map(n_true).fillna(0).astype(int)
    F.to_parquet(args.out)
    # entities with no candidates at all still count in macro F0.5 - keep the list
    s1[["entity_id"]].merge(n_true.rename("n_true_total"), left_on="entity_id",
                            right_index=True, how="left").fillna(0).to_parquet(
        args.out.replace(".parquet", "_entities.parquet"))
    log(f"saved {len(F)} pairs, positives={F['label'].sum()} "
        f"(blocking recall {F['label'].sum() / max(1, len(truth)):.4f})")


if __name__ == "__main__":
    main()
