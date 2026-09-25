"""Build the train Source 2/3 blocking index without running the recall evaluation.

Usage: python src/blocking/build_train_index.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import TRAIN_DIR, TRAIN_INDEX
from blocking.tfidf_blocking import build_index

if __name__ == "__main__":
    if os.path.exists(os.path.join(TRAIN_INDEX, "countries.json")):
        print("train index already built")
    else:
        build_index([os.path.join(TRAIN_DIR, f"train_source{i}.tsv") for i in (2, 3)], TRAIN_INDEX)
