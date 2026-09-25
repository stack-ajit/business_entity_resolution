"""
Pair features for the matching model: given candidate pairs (S1 record, S2/S3 record)
from blocking, describe how similar the two records are.

Design notes:
- Country is deliberately NOT a feature: the test set contains France, unseen in
  training. Features are country-agnostic similarities, so the model transfers.
- Retrieval context (score, rank, score relative to the best candidate of the same
  S1 entity) is included: "how good is this candidate compared to the others" is a
  strong signal for precision-heavy F0.5.
"""
import itertools
import multiprocessing as mp
import os
import re
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from blocking.tfidf_blocking import normalize_tokens, skeleton, read_tsv_chunks

# Legal / generic suffixes: compared separately so "X Pvt Ltd" vs "X" is not penalised
# on the core name. Includes France forms; unknown suffixes just stay in the core name.
LEGAL = {
    "pvt", "private", "ltd", "limited", "llc", "inc", "incorporated", "co", "company",
    "corp", "corporation", "llp", "pllc", "lp", "plc", "pc", "pa", "the", "and", "of",
    "sas", "sarl", "sa", "sasu", "eurl", "sci", "com", "www", "dba", "formerly",
}
_digits = re.compile(r"\d+")


def _prep(name, address):
    nt = normalize_tokens(name)
    core = [t for t in nt if t not in LEGAL]
    at = normalize_tokens(address)
    nums = {t.lstrip("0") or "0" for t in at if t.isdigit()}
    return (" ".join(nt), " ".join(core), " ".join(skeleton(t) for t in core),
            "".join(core), " ".join(at), nums, set(core), set(at))


def prepare(name, address):
    """Prepared tuple used by pair_features (normalized strings, sets, flags)."""
    return _prep(name, address) + (bool(name) and not name.isascii(), address == "")


def load_records(paths, ids=None):
    """Load records (optionally only the given entity ids) as {id: prepared tuple}."""
    ids = None if ids is None else set(ids)
    out = {}
    for p in paths:
        for ch in read_tsv_chunks(p, 500_000):
            if ids is not None:
                ch = ch[ch["entity_id"].isin(ids)]
            for i, n, a in zip(ch["entity_id"].values, ch["business_name"].values,
                               ch["business_address"].values):
                out[i] = prepare(n, a)
    return out


def load_raw(paths):
    """{id: (name, address)} for all records - read once, prepare lazily per batch."""
    out = {}
    for p in paths:
        for ch in read_tsv_chunks(p, 500_000):
            out.update(zip(ch["entity_id"].values,
                           zip(ch["business_name"].values, ch["business_address"].values)))
    return out


def _jacc(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


FEATURES = [
    "score", "rank", "score_rel", "score_gap_best", "n_cands",
    "name_ratio", "name_tset", "name_tsort", "name_partial", "name_jw",
    "core_ratio", "core_tset", "core_partial", "core_concat_ratio", "skel_ratio", "skel_tset",
    "core_jacc", "name_len_a", "name_len_b",
    "addr_ratio", "addr_tset", "addr_partial", "addr_jacc",
    "num_jacc", "num_shared", "num_conflict", "num_a", "num_b", "max_shared_num_len",
    "postcode_match", "postcode_conflict",
    "addr_empty_b", "addr_empty_a", "name_nonascii_b", "is_s3",
]


def _rows(payload):
    """
    Similarity features for a chunk of pairs. payload = list of (A, B, is_s3) where A/B are
    prepared record tuples. Returns a compact float32 array (n, n_features): a list of
    Python float tuples would cost ~1 KB per pair (~7 GB for 7.5M pairs).
    """
    out = np.empty((len(payload), len(FEATURES) - 5), dtype=np.float32)
    for k, (A, B, is_s3) in enumerate(payload):
        na, nb = A[5], B[5]
        shared = na & nb
        pa = {x for x in na if len(x) in (5, 6)}
        pb = {x for x in nb if len(x) in (5, 6)}
        out[k] = (
            fuzz.ratio(A[0], B[0]), fuzz.token_set_ratio(A[0], B[0]),
            fuzz.token_sort_ratio(A[0], B[0]), fuzz.partial_ratio(A[0], B[0]),
            JaroWinkler.similarity(A[0], B[0]),
            fuzz.ratio(A[1], B[1]), fuzz.token_set_ratio(A[1], B[1]), fuzz.partial_ratio(A[1], B[1]),
            fuzz.ratio(A[3], B[3]), fuzz.ratio(A[2], B[2]), fuzz.token_set_ratio(A[2], B[2]),
            _jacc(A[6], B[6]), len(A[6]), len(B[6]),
            fuzz.ratio(A[4], B[4]), fuzz.token_set_ratio(A[4], B[4]), fuzz.partial_ratio(A[4], B[4]),
            _jacc(A[7], B[7]),
            _jacc(na, nb), len(shared), float(bool(na) and bool(nb) and not shared), len(na), len(nb),
            max((len(x) for x in shared), default=0),
            float(bool(pa & pb)), float(bool(pa) and bool(pb) and not (pa & pb)),
            float(B[9]), float(A[9]), float(B[8]), float(is_s3),
        )
    return out


def _payloads(a_ids, b_ids, s1_rec, s23_rec, chunk):
    for i in range(0, len(a_ids), chunk):
        yield [(s1_rec[a], s23_rec[b], b.startswith("S3"))
               for a, b in zip(a_ids[i:i + chunk], b_ids[i:i + chunk])]


def pair_features(cands, s1_rec, s23_rec, n_jobs=None, chunk=20_000):
    """
    cands: DataFrame with source1_entity_id, candidate_entity_id, score. Returns features.
    n_jobs: worker processes for the string similarities (default: all cores). Workers get
    only the small chunk of record tuples they score (never the big dicts), so memory stays
    flat; results come back as float32 arrays. Falls back to 1 process on any pool error.
    """
    c = cands.copy()
    g = c.groupby("source1_entity_id")["score"]
    c["rank"] = g.rank(ascending=False, method="first")
    best = g.transform("max")
    c["score_rel"] = c["score"] / best
    c["score_gap_best"] = best - c["score"]
    c["n_cands"] = g.transform("size")

    a_ids = c["source1_entity_id"].values
    b_ids = c["candidate_entity_id"].values
    n_jobs = n_jobs or os.cpu_count() or 1
    parts = None
    if n_jobs > 1 and len(c) > 4 * chunk:
        try:
            parts = []
            gen = _payloads(a_ids, b_ids, s1_rec, s23_rec, chunk)
            with mp.Pool(n_jobs) as pool:
                # bounded waves: Pool.imap would drain the generator into its queue at once
                while wave := list(itertools.islice(gen, n_jobs * 2)):
                    parts += pool.map(_rows, wave, chunksize=1)
        except Exception as e:  # never lose a long run to a pool problem
            print(f"parallel features failed ({e!r}), falling back to 1 process", flush=True)
            parts = None
    if parts is None:
        parts = [_rows(p) for p in _payloads(a_ids, b_ids, s1_rec, s23_rec, chunk)]
    X = np.vstack(parts) if parts else np.empty((0, len(FEATURES) - 5), np.float32)
    F = pd.DataFrame(X, columns=FEATURES[5:], index=c.index)
    return pd.concat([c[["source1_entity_id", "candidate_entity_id"] + FEATURES[:5]], F], axis=1)
