"""
End-to-end test inference: blocking -> pair features -> LightGBM -> selection -> TSVs.

Streams Source 1 in batches so ~86M candidate pairs (1.7M S1 x top-50) never sit in
memory at once. Writes, per the challenge spec:
    output/candidate_pairs.tsv   every candidate the model scored (the exact model input)
    output/matching_results.tsv  the selected matches (subset of candidates)
One row per Source 1 entity in both files; empty list when there is nothing.

Usage: python src/matching/predict_test.py [--top-k 50] [--batch 100000]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import ROOT, TEST_DIR, TEST_INDEX, CACHE_DIR, OUTPUT_DIR
from blocking.tfidf_blocking import build_index, query_index, read_tsv_chunks
from matching.pair_features import FEATURES, load_raw, pair_features, prepare
from matching.selection import select

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--batch", type=int, default=100_000)
    args = ap.parse_args()

    s23_paths = [os.path.join(TEST_DIR, f"test_source{i}.tsv") for i in (2, 3)]
    if not os.path.exists(os.path.join(TEST_INDEX, "countries.json")):
        build_index(s23_paths, TEST_INDEX, log=log)

    model_dir = os.path.join(CACHE_DIR, "model")
    model = lgb.Booster(model_file=os.path.join(model_dir, "lgb_matcher.txt"))
    with open(os.path.join(model_dir, "selection.json")) as f:
        sel = json.load(f)
    log(f"selection rule: {sel}")

    s1 = pd.concat(read_tsv_chunks(os.path.join(TEST_DIR, "test_source1.tsv")), ignore_index=True)
    raw = load_raw(s23_paths)
    log(f"{len(s1)} test S1 entities, {len(raw)} S2/S3 records")

    # Each finished batch is checkpointed, so a killed session resumes where it stopped.
    ckpt_dir = os.path.join(CACHE_DIR, f"test_batches_k{args.top_k}_b{args.batch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    for b0 in range(0, len(s1), args.batch):
        ckpt = os.path.join(ckpt_dir, f"batch_{b0:08d}.parquet")
        if os.path.exists(ckpt):
            continue
        t = [time.time()]
        b = s1.iloc[b0:b0 + args.batch]
        cands = query_index(b["entity_id"].values, b["business_name"].values,
                            b["business_address"].values, b["country"].values, TEST_INDEX,
                            top_k=args.top_k, log=lambda m: None)
        t.append(time.time())
        out = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_ids", "matched_entity_ids"])
        if len(cands):
            s1_rec = {i: prepare(n, a) for i, n, a in
                      zip(b["entity_id"].values, b["business_name"].values, b["business_address"].values)}
            s23_rec = {i: prepare(*raw[i]) for i in cands["candidate_entity_id"].unique()}
            F = pair_features(cands, s1_rec, s23_rec)
            t.append(time.time())
            F["p"] = model.predict(F[FEATURES], num_threads=os.cpu_count())
            t.append(time.time())
            kept = select(F, sel["threshold"], sel["rel"], sel["max_matches"])
            out = pd.concat([
                F.groupby("source1_entity_id")["candidate_entity_id"].agg(",".join).rename("candidate_entity_ids"),
                kept.groupby("source1_entity_id")["candidate_entity_id"].agg(",".join).rename("matched_entity_ids"),
            ], axis=1).fillna("").rename_axis("source1_entity_id").reset_index()
        out.to_parquet(ckpt)
        steps = np.diff(t).round().astype(int).tolist()
        log(f"batch {b0}-{b0 + len(b)}: {len(cands)} candidates, "
            f"{(out['matched_entity_ids'] != '').sum()} entities with matches "
            f"[secs query/features/predict: {steps}]")

    done = pd.concat([pd.read_parquet(os.path.join(ckpt_dir, f)) for f in sorted(os.listdir(ckpt_dir))],
                     ignore_index=True).set_index("source1_entity_id")
    cand_lists, match_lists = done["candidate_entity_ids"], done["matched_entity_ids"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ids = s1["entity_id"]
    pd.DataFrame({"source1_entity_id": ids, "candidate_entity_ids": ids.map(cand_lists).fillna("")}).to_csv(
        os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": ids, "matched_entity_ids": ids.map(match_lists).fillna("")}).to_csv(
        os.path.join(OUTPUT_DIR, "matching_results.tsv"), sep="\t", index=False)
    n_empty = (ids.map(match_lists).fillna("") == "").mean()
    log(f"wrote outputs to {OUTPUT_DIR}; {n_empty:.2%} of entities predicted as singletons")

    validator = os.path.join(ROOT, "student_resource", "utils", "validate_submission.py")
    if os.path.exists(validator):
        subprocess.run([sys.executable, validator,
                        "--matching", os.path.join(OUTPUT_DIR, "matching_results.tsv"),
                        "--candidate", os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
                        "--test-dir", TEST_DIR], check=False)


if __name__ == "__main__":
    main()
