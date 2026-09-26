"""
Central paths so the same code runs locally and on Kaggle.

    ER_DATA_DIR  folder containing train/ and test/   (read-only is fine)
    ER_WORK_DIR  writable folder for caches, models, outputs

Local defaults: student_resource/dataset and data/.
Kaggle example:  ER_DATA_DIR=/kaggle/input/<dataset-slug>  ER_WORK_DIR=/kaggle/working
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get("ER_DATA_DIR", os.path.join(ROOT, "student_resource", "dataset"))
WORK_DIR = os.environ.get("ER_WORK_DIR", os.path.join(ROOT, "data"))

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_DIR = os.path.join(WORK_DIR, "sample")
CACHE_DIR = os.path.join(WORK_DIR, "cache")
TRAIN_INDEX = os.path.join(CACHE_DIR, "train_index")        # forward: S2/S3 records
TEST_INDEX = os.path.join(CACHE_DIR, "test_index")
TRAIN_S1_INDEX = os.path.join(CACHE_DIR, "train_s1_index")  # reverse: S1 records
TEST_S1_INDEX = os.path.join(CACHE_DIR, "test_s1_index")
TRAIN_REVERSE = os.path.join(CACHE_DIR, "train_reverse.parquet")  # each S2/S3 -> its top S1s
TEST_REVERSE = os.path.join(CACHE_DIR, "test_reverse.parquet")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
