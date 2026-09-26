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
from matching.pair_features import (add_context_features, load_raw, pair_features, prepare,
                                    sibling_expand, union_reverse_candidates)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=150_000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-s1-frac", type=float, default=0.18,
                    help="share of train S1 removed from the world to match test's distractor rate")
    ap.add_argument("--block", type=int, default=20_000, help="S1 entities per feature block (memory)")
    ap.add_argument("--out", default=os.path.join(CACHE, "train_pairs_v4.parquet"))
    args = ap.parse_args()

    s1_path = os.path.join(TRAIN, "train_source1.tsv")
    s23_paths = [os.path.join(TRAIN, f"train_source{i}.tsv") for i in (2, 3)]

    # Make the train world look like test: test has ~2.3 distractor S2/S3 records per S1
    # vs ~1.2 in train. Removing a fraction of S1 entities entirely turns their matched
    # records into distractors that belong to nobody (exactly the test situation).
    s1_all = pd.concat(read_tsv_chunks(s1_path), ignore_index=True)
    s1_all = s1_all.sample(frac=1 - args.drop_s1_frac, random_state=args.seed + 1)
    kept_path = os.path.join(CACHE, "train_source1_kept.tsv")
    if not os.path.exists(kept_path):
        s1_all.to_csv(kept_path, sep="\t", index=False)
    log(f"S1 world: kept {len(s1_all)} entities (dropped {args.drop_s1_frac:.0%} -> their matches become distractors)")
    build_s1_index(kept_path, TRAIN_S1_INDEX, log=log)
    rev = build_reverse_table(s23_paths, TRAIN_S1_INDEX, TRAIN_REVERSE, log=log)
    rev_sum = reverse_summary(rev)
    log(f"reverse table: {len(rev)} rows for {len(rev_sum)} S2/S3 records")

    s1 = s1_all.sample(n=args.n_s1, random_state=args.seed).reset_index(drop=True)
    del s1_all
    log(f"sampled {len(s1)} S1 entities")

    cands = query_index(s1["entity_id"].values, s1["business_name"].values,
                        s1["business_address"].values, s1["country"].values,
                        TRAIN_INDEX, top_k=args.top_k, log=log)
    n_fwd = len(cands)
    cands = union_reverse_candidates(cands, rev, s1["entity_id"].values)
    log(f"{n_fwd} forward pairs + {len(cands) - n_fwd} reverse-only pairs")
    s23_raw = load_raw(s23_paths)  # raw strings of all S2/S3 (anchors + expansion + features)
    n_before = len(cands)
    cands, edges = sibling_expand(cands, rev, s23_raw, TRAIN_INDEX)
    log(f"sibling expansion: +{len(cands) - n_before} sibling-only pairs")
    ctx = add_context_features(cands, rev, rev_sum)
    del rev, rev_sum

    gt = pd.concat(read_tsv_chunks(os.path.join(TRAIN, "train_ground_truth.tsv")), ignore_index=True)
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
    truth = {(a, b) for a, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
             for b in m.split(",") if b}
    n_true = gt.set_index("source1_entity_id")["matched_entity_ids"].map(
        lambda m: len([x for x in m.split(",") if x]))

    # Memory: prepared records (strings, sets, IDF dicts) cost ~2-3 KB each, so preparing all
    # ~6M candidate records at once exhausts 30 GB. Keep only raw strings, and prepare +
    # score in blocks of S1 entities, discarding each block's prepared records afterwards.
    idf = load_idf(TRAIN_INDEX)
    s1_raw = load_raw([s1_path], s1["entity_id"])
    log(f"computing features in blocks for {len(ctx)} pairs")
    ids = s1["entity_id"].values
    parts = []
    for i in range(0, len(ids), args.block):
        blk = ctx[ctx["source1_entity_id"].isin(set(ids[i:i + args.block]))]
        s1_rec = {a: prepare(*s1_raw[a][:2], idf.get(s1_raw[a][2])) for a in blk["source1_entity_id"].unique()}
        s23_rec = {b: prepare(*s23_raw[b][:2], idf.get(s23_raw[b][2])) for b in blk["candidate_entity_id"].unique()}
        parts.append(pair_features(blk, s1_rec, s23_rec))
        del s1_rec, s23_rec
        log(f"features: {min(i + args.block, len(ids))}/{len(ids)} S1 entities")
    F = pd.concat(parts, ignore_index=True)
    del parts, ctx
    F["label"] = [int((a, b) in truth) for a, b in zip(F["source1_entity_id"], F["candidate_entity_id"])]
    F["n_true_total"] = F["source1_entity_id"].map(n_true).fillna(0).astype(int)
    F.to_parquet(args.out)
    edges.to_parquet(args.out.replace(".parquet", "_edges.parquet"))
    # entities with no candidates at all still count in macro F0.5 - keep the list
    s1[["entity_id"]].merge(n_true.rename("n_true_total"), left_on="entity_id",
                            right_index=True, how="left").fillna(0).to_parquet(
        args.out.replace(".parquet", "_entities.parquet"))
    pos = F["label"] == 1
    sib_only = (F["fwd"] == 0) & (F["rscore"] == 0)
    log(f"saved {len(F)} pairs, positives={pos.sum()}: blocking recall forward-only "
        f"{(pos & (F['fwd'] == 1)).sum() / len(truth):.4f}, + reverse "
        f"{(pos & ~sib_only).sum() / len(truth):.4f}, + siblings {pos.sum() / len(truth):.4f}")


if __name__ == "__main__":
    main()
