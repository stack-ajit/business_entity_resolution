"""
One-command end-to-end pipeline: data -> blocking -> training pairs -> two-stage matcher
-> test inference -> output/matching_results.tsv + output/candidate_pairs.tsv.

Each stage is skipped if its output already exists, so a rerun resumes where it stopped.
Paths come from src/config.py (env vars ER_DATA_DIR = folder with train/ and test/,
ER_WORK_DIR = writable folder for caches and outputs).

Usage:
    python run_pipeline.py                 # full pipeline
    python run_pipeline.py --eval-blocking # also report blocking recall on full train
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config import CACHE_DIR, OUTPUT_DIR, TRAIN_INDEX  # noqa: E402


def run(title, script, *args, done=None):
    """Run one stage as a subprocess unless its output file already exists."""
    if done and os.path.exists(done):
        print(f"== {title}: already done ({done})", flush=True)
        return
    print(f"== {title}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, "-u", os.path.join(ROOT, script), *args], check=True)
    print(f"== {title}: finished in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-blocking", action="store_true")
    ap.add_argument("--n-s1", default="150000")
    ap.add_argument("--top-k", default="50")
    ap.add_argument("--stage1-recall", default="0.998")
    args = ap.parse_args()

    if args.eval_blocking:
        run("sample for blocking evaluation", "src/data/create_sample.py")
        run("blocking recall on full train", "src/blocking/evaluate_blocking.py", "--top-k", "100")
    run("train S2/S3 blocking index", "src/blocking/build_train_index.py",
        done=os.path.join(TRAIN_INDEX, "countries.json"))
    run("labelled training pairs", "src/matching/build_training_set.py",
        "--n-s1", args.n_s1, "--top-k", args.top_k,
        done=os.path.join(CACHE_DIR, "train_pairs_v2.parquet"))
    run("two-stage matcher", "src/matching/train_matcher.py", "--stage1-recall", args.stage1_recall,
        done=os.path.join(CACHE_DIR, "model_v2", "stage2.txt"))
    run("test inference", "src/matching/predict_test.py", "--top-k", args.top_k)
    print(f"outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
