"""
Build the labelled pair dataset for the matching model.

Takes a random subset of train Source 1 entities, runs the SAME candidate generation used
at test time against the full train data, labels each candidate with the ground truth, and
computes all features. Training on real blocking output (not random negatives) means the
model learns to reject exactly the look-alikes it will face.

Candidate generation = forward top-K (S1 -> S2/S3)  UNION  reverse top-R (S2/S3 -> S1).
The reverse table is computed against ALL train S1 entities, exactly as at test time
(where every test S1 is present), so reverse features mean the same thing in both.

Usage:
    python src/matching/build_training_set.py --n-s1 150000 --top-k 50
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import TRAIN_DIR as TRAIN, CACHE_DIR as CACHE, TRAIN_INDEX, TRAIN_S1_INDEX, TRAIN_REVERSE
from blocking.tfidf_blocking import load_idf, query_index, read_tsv_chunks
from blocking.reverse import build_reverse_table, build_s1_index, reverse_summary
from matching.pair_features import (add_context_features, load_records, pair_features,
                                    union_reverse_candidates)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=150_000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(CACHE, "train_pairs_v2.parquet"))
    args = ap.parse_args()

    s1_path = os.path.join(TRAIN, "train_source1.tsv")
    s23_paths = [os.path.join(TRAIN, f"train_source{i}.tsv") for i in (2, 3)]
    build_s1_index(s1_path, TRAIN_S1_INDEX, log=log)
    rev = build_reverse_table(s23_paths, TRAIN_S1_INDEX, TRAIN_REVERSE, log=log)
    rev_sum = reverse_summary(rev)
    log(f"reverse table: {len(rev)} rows for {len(rev_sum)} S2/S3 records")

    s1 = pd.concat(read_tsv_chunks(s1_path), ignore_index=True)
    s1 = s1.sample(n=args.n_s1, random_state=args.seed).reset_index(drop=True)
    log(f"sampled {len(s1)} S1 entities")

    cands = query_index(s1["entity_id"].values, s1["business_name"].values,
                        s1["business_address"].values, s1["country"].values,
                        TRAIN_INDEX, top_k=args.top_k, log=log)
    n_fwd = len(cands)
    cands = union_reverse_candidates(cands, rev, s1["entity_id"].values)
    log(f"{n_fwd} forward pairs + {len(cands) - n_fwd} reverse-only pairs")
    ctx = add_context_features(cands, rev, rev_sum)
    del rev, rev_sum

    gt = pd.concat(read_tsv_chunks(os.path.join(TRAIN, "train_ground_truth.tsv")), ignore_index=True)
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
    truth = {(a, b) for a, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
             for b in m.split(",") if b}
    n_true = gt.set_index("source1_entity_id")["matched_entity_ids"].map(
        lambda m: len([x for x in m.split(",") if x]))

    idf = load_idf(TRAIN_INDEX)
    s1_rec = load_records([s1_path], s1["entity_id"], idf)
    s23_rec = load_records(s23_paths, ctx["candidate_entity_id"].unique(), idf)
    log("records loaded, computing features")
    F = pair_features(ctx, s1_rec, s23_rec)
    F["label"] = [int((a, b) in truth) for a, b in zip(F["source1_entity_id"], F["candidate_entity_id"])]
    F["n_true_total"] = F["source1_entity_id"].map(n_true).fillna(0).astype(int)
    F.to_parquet(args.out)
    # entities with no candidates at all still count in macro F0.5 - keep the list
    s1[["entity_id"]].merge(n_true.rename("n_true_total"), left_on="entity_id",
                            right_index=True, how="left").fillna(0).to_parquet(
        args.out.replace(".parquet", "_entities.parquet"))
    pos = F["label"] == 1
    log(f"saved {len(F)} pairs, positives={pos.sum()}: blocking recall forward-only "
        f"{(pos & (F['fwd'] == 1)).sum() / len(truth):.4f}, with reverse {pos.sum() / len(truth):.4f}")


if __name__ == "__main__":
    main()
