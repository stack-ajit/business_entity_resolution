"""
Reverse retrieval: for every Source 2/3 record, its top-R most similar Source 1 entities.

Why: the ground truth is strictly one-to-one - each S2/S3 record belongs to at most one
S1 entity (7,638,365 matched ids in train, all unique). Forward retrieval (S1 -> S2/S3)
cannot see that a look-alike candidate is an even better fit for a *different* S1
entity. The reverse view gives, for each candidate, "which S1 does this record itself
point to, and how decisively" - both as model features and as extra candidates.

Ids are stored as int64 codes (S1-123 -> 123, S2-123 -> 123, S3-123 -> 123 + 1e10) so a
10M-row reverse table stays well under 1 GB.
"""
import os
import time

import numpy as np
import pandas as pd

from blocking.tfidf_blocking import build_index, query_index, read_tsv_chunks

S3_OFFSET = 10_000_000_000


def id_code(ids):
    """Vectorised 'S1-123'/'S2-123' -> 123, 'S3-123' -> 123 + 1e10 (int64)."""
    s = pd.Series(np.asarray(ids, dtype=object))
    num = s.str[3:].astype(np.int64).to_numpy()
    return num + np.where(s.str[:2].to_numpy() == "S3", S3_OFFSET, 0).astype(np.int64)


def decode_s23(codes):
    codes = np.asarray(codes, dtype=np.int64)
    s3 = codes >= S3_OFFSET
    return np.where(s3, "S3-", "S2-").astype(object) + np.where(s3, codes - S3_OFFSET, codes).astype(str).astype(object)


def decode_s1(codes):
    return "S1-" + np.asarray(codes, dtype=np.int64).astype(str).astype(object)


def build_s1_index(s1_path, index_dir, log=print):
    if not os.path.exists(os.path.join(index_dir, "countries.json")):
        build_index([s1_path], index_dir, log=log)


def build_reverse_table(s23_paths, s1_index_dir, out_path, top_r=3, chunk=500_000, log=print):
    """Query every S2/S3 record against the S1 index; save (b, s1, rscore, rrank) as parquet."""
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)
    parts, t0, n = [], time.time(), 0
    for path in s23_paths:
        for ch in read_tsv_chunks(path, chunk):
            r = query_index(ch["entity_id"].values, ch["business_name"].values,
                            ch["business_address"].values, ch["country"].values, s1_index_dir,
                            top_k=top_r, batch_size=chunk, log=lambda m: None)
            parts.append(pd.DataFrame({
                "b": id_code(r["source1_entity_id"].values),
                "s1": id_code(r["candidate_entity_id"].values),
                "rscore": r["score"].to_numpy(np.float32),
            }))
            n += len(ch)
            log(f"reverse: {n} S2/S3 records queried ({time.time() - t0:.0f}s)")
    rev = pd.concat(parts, ignore_index=True)
    rev["rrank"] = rev.groupby("b")["rscore"].rank(ascending=False, method="first").astype(np.int8)
    rev.to_parquet(out_path)
    return rev


def reverse_summary(rev):
    """Per S2/S3 record: best and second-best S1 score (how decisively it points somewhere)."""
    best = rev.loc[rev["rrank"] == 1, ["b", "rscore"]].set_index("b")["rscore"].rename("rev_best")
    second = rev.loc[rev["rrank"] == 2, ["b", "rscore"]].set_index("b")["rscore"].rename("rev_second")
    return pd.concat([best, second], axis=1).fillna(0.0).astype(np.float32)
