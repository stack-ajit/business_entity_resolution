"""
End-to-end test inference:
  forward blocking (top-K) + reverse candidates -> context features -> stage-1 pruner
  -> string features on survivors -> stage-2 matcher -> global one-to-one resolution
  -> per-entity selection -> TSVs.

Streams Source 1 in checkpointed batches (a killed session resumes where it stopped).
Writes, per the challenge spec:
    output/candidate_pairs.tsv   stage-1 survivors = exactly the pairs the matcher scored
    output/matching_results.tsv  the selected matches (subset of candidates)
One row per Source 1 entity in both files; empty list when there is nothing.

One-to-one resolution needs every S1's claims, so it runs after all batches: each S2/S3
record keeps only its highest-probability S1 (true structure of the data).

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
from config import ROOT, TEST_DIR, TEST_INDEX, TEST_S1_INDEX, TEST_REVERSE, CACHE_DIR, OUTPUT_DIR
from blocking.tfidf_blocking import build_index, load_idf, query_index, read_tsv_chunks
from blocking.reverse import build_reverse_table, build_s1_index, reverse_summary
from matching.pair_features import (add_context_features, load_raw, pair_features, prepare,
                                    second_order_features, sibling_expand, union_reverse_candidates)
from matching.selection import select, select_expected

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--batch", type=int, default=100_000)
    ap.add_argument("--model-dir", default=os.path.join(CACHE_DIR, "model_v4"))
    args = ap.parse_args()

    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    s23_paths = [os.path.join(TEST_DIR, f"test_source{i}.tsv") for i in (2, 3)]
    if not os.path.exists(os.path.join(TEST_INDEX, "countries.json")):
        build_index(s23_paths, TEST_INDEX, log=log)
    build_s1_index(s1_path, TEST_S1_INDEX, log=log)
    rev = build_reverse_table(s23_paths, TEST_S1_INDEX, TEST_REVERSE, log=log)
    rev_sum = reverse_summary(rev)
    log(f"reverse table: {len(rev)} rows")

    m1 = lgb.Booster(model_file=os.path.join(args.model_dir, "stage1.txt"))
    m2 = lgb.Booster(model_file=os.path.join(args.model_dir, "stage2.txt"))
    with open(os.path.join(args.model_dir, "selection.json")) as f:
        sel = json.load(f)
    # select columns by the names stored in each model, so code and model can't drift apart
    f1, f2 = m1.feature_name(), m2.feature_name()
    use_sib = "sib_n" in f1
    m3 = None
    if sel.get("use_stage3"):
        m3 = lgb.Booster(model_file=os.path.join(args.model_dir, "stage3.txt"))
    log(f"selection rule: {sel}; sibling expansion: {use_sib}; stage 3: {m3 is not None}")

    idf = load_idf(TEST_INDEX)
    s1 = pd.concat(read_tsv_chunks(s1_path), ignore_index=True)
    raw = load_raw(s23_paths)
    log(f"{len(s1)} test S1 entities, {len(raw)} S2/S3 records")

    ckpt_dir = os.path.join(CACHE_DIR, f"test_batches_{os.path.basename(args.model_dir)}_k{args.top_k}_b{args.batch}")
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
        cands = union_reverse_candidates(cands, rev, b["entity_id"].values)
        edges = None
        if use_sib:
            cands, edges = sibling_expand(cands, rev, raw, TEST_INDEX)
        ctx = add_context_features(cands, rev, rev_sum)
        t.append(time.time())
        ctx = ctx[m1.predict(ctx[f1], num_threads=os.cpu_count()) >= sel["stage1_threshold"]]
        ctx = ctx.reset_index(drop=True)
        t.append(time.time())
        out = pd.DataFrame({"source1_entity_id": pd.Series(dtype=object),
                            "candidate_entity_id": pd.Series(dtype=object),
                            "p": pd.Series(dtype=np.float32)})
        if len(ctx):
            s1_rec = {i: prepare(n, a, idf.get(c)) for i, n, a, c in
                      zip(b["entity_id"].values, b["business_name"].values,
                          b["business_address"].values, b["country"].values)}
            s23_rec = {i: prepare(raw[i][0], raw[i][1], idf.get(raw[i][2]))
                       for i in ctx["candidate_entity_id"].unique()}
            F = pair_features(ctx, s1_rec, s23_rec)
            t.append(time.time())
            p = m2.predict(F[f2], num_threads=os.cpu_count())
            if m3 is not None:
                # stage 3: entity context + sibling consistency from stage-2 probabilities.
                # All candidates of an S1 are in the same batch, so the context is complete.
                F["p2"] = p
                F = second_order_features(F, edges)
                p = m3.predict(F[m3.feature_name()], num_threads=os.cpu_count())
            out = pd.DataFrame({"source1_entity_id": F["source1_entity_id"].values,
                                "candidate_entity_id": F["candidate_entity_id"].values,
                                "p": p.astype(np.float32)})
            t.append(time.time())
        out.to_parquet(ckpt)
        steps = np.diff(t).round().astype(int).tolist()
        log(f"batch {b0}-{b0 + len(b)}: {len(cands)} raw candidates -> {len(out)} after stage 1 "
            f"({len(out) / len(b):.2f} per S1) [secs retrieve+context/stage1/features/stage2: {steps}]")

    allp = pd.concat([pd.read_parquet(os.path.join(ckpt_dir, f)) for f in sorted(os.listdir(ckpt_dir))],
                     ignore_index=True)
    if sel.get("mode") == "expected":
        kept = select_expected(allp, sel["miss"], sel["power"])
    else:
        kept = select(allp, sel["threshold"], sel["rel"], sel["max_matches"], one_to_one=True)
    n_before = (allp["p"] >= sel["threshold"]).sum()
    log(f"{len(allp)} scored pairs; {n_before} above threshold -> {len(kept)} after one-to-one + rules")

    cand_lists = allp.groupby("source1_entity_id")["candidate_entity_id"].agg(",".join)
    match_lists = kept.groupby("source1_entity_id")["candidate_entity_id"].agg(",".join)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ids = s1["entity_id"]
    pd.DataFrame({"source1_entity_id": ids, "candidate_entity_ids": ids.map(cand_lists).fillna("")}).to_csv(
        os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": ids, "matched_entity_ids": ids.map(match_lists).fillna("")}).to_csv(
        os.path.join(OUTPUT_DIR, "matching_results.tsv"), sep="\t", index=False)
    n_empty = (ids.map(match_lists).fillna("") == "").mean()
    log(f"wrote outputs to {OUTPUT_DIR}; {len(allp) / len(ids):.2f} candidates per S1; "
        f"{len(kept) / len(ids):.2f} matches per S1; {n_empty:.2%} predicted singletons")

    validator = os.path.join(ROOT, "student_resource", "utils", "validate_submission.py")
    if os.path.exists(validator):
        subprocess.run([sys.executable, validator,
                        "--matching", os.path.join(OUTPUT_DIR, "matching_results.tsv"),
                        "--candidate", os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
                        "--test-dir", TEST_DIR], check=False)


if __name__ == "__main__":
    main()
