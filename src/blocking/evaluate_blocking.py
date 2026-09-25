"""
Measure blocking recall honestly: sampled Source 1 queries vs the FULL train Source 2/3
(10.3M records), not the 134K-row sample, whose small noise pool flatters any method.

Usage:
    python src/blocking/evaluate_blocking.py            # build index (once) + evaluate
    python src/blocking/evaluate_blocking.py --top-k 100 --min-idf 5.5
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from blocking.tfidf_blocking import build_index, query_index

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN = os.path.join(ROOT, "student_resource", "dataset", "train")
SAMPLE = os.path.join(ROOT, "data", "sample")
CACHE = os.path.join(ROOT, "data", "cache", "train_index")

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--min-idf", type=float, default=None)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.rebuild or not os.path.exists(os.path.join(CACHE, "countries.json")):
        build_index([os.path.join(TRAIN, f"train_source{i}.tsv") for i in (2, 3)], CACHE, min_idf=args.min_idf, log=log)

    s1 = pd.read_csv(os.path.join(SAMPLE, "sample_source1.tsv"), sep="\t", dtype=str,
                     keep_default_na=False)
    gt = pd.read_csv(os.path.join(SAMPLE, "sample_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False)
    cands = query_index(s1["entity_id"].values, s1["business_name"].values,
                        s1["business_address"].values, s1["country"].values, CACHE,
                        top_k=args.top_k, log=log)

    truth = gt[gt["matched_entity_ids"] != ""].assign(
        candidate_entity_id=lambda d: d["matched_entity_ids"].str.split(",")).explode("candidate_entity_id")
    truth = truth.rename(columns={"source1_entity_id": "source1_entity_id"})[
        ["source1_entity_id", "candidate_entity_id"]]
    n_true = len(truth)

    cands["rank"] = cands.groupby("source1_entity_id")["score"].rank(ascending=False, method="first")
    hit = cands.merge(truth.assign(is_true=1), how="left").fillna({"is_true": 0})

    print(f"\ntrue pairs: {n_true}   queries: {len(s1)}")
    print("recall@K  (candidates per S1 = K)")
    for k in [5, 10, 20, 30, 50, 75, 100, 150, 200]:
        if k > args.top_k:
            break
        sub = hit[hit["rank"] <= k]
        print(f"  K={k:4d}  recall={sub['is_true'].sum() / n_true:.4f}  pairs={len(sub)}")
    # adaptive cut: keep candidates scoring >= frac * best score for that S1
    best = hit.groupby("source1_entity_id")["score"].transform("max")
    for frac in [0.3, 0.4, 0.5, 0.6]:
        sub = hit[hit["score"] >= frac * best]
        print(f"  score>={frac:.1f}*best  recall={sub['is_true'].sum() / n_true:.4f}  pairs={len(sub)}")

    missed = truth.merge(cands, how="left")
    missed = missed[missed["score"].isna()]
    by_country = truth.merge(s1[["entity_id", "country"]], left_on="source1_entity_id",
                             right_on="entity_id")
    found = set(zip(hit.loc[hit.is_true == 1, "source1_entity_id"], hit.loc[hit.is_true == 1, "candidate_entity_id"]))
    by_country["found"] = [(a, b) in found for a, b in zip(by_country.source1_entity_id, by_country.candidate_entity_id)]
    print("recall by country @top_k:", by_country.groupby("country")["found"].mean().round(4).to_dict())
    by_country["src"] = by_country.candidate_entity_id.str[:2]
    print("recall by source  @top_k:", by_country.groupby("src")["found"].mean().round(4).to_dict())
    os.makedirs(os.path.join(ROOT, "data", "cache"), exist_ok=True)
    missed.to_csv(os.path.join(ROOT, "data", "cache", "missed_pairs.tsv"), sep="\t", index=False)
    log("done")


if __name__ == "__main__":
    main()
