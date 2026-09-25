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


def pair_features(cands, s1_rec, s23_rec):
    """cands: DataFrame with source1_entity_id, candidate_entity_id, score. Returns features."""
    c = cands.copy()
    g = c.groupby("source1_entity_id")["score"]
    c["rank"] = g.rank(ascending=False, method="first")
    best = g.transform("max")
    c["score_rel"] = c["score"] / best
    c["score_gap_best"] = best - c["score"]
    c["n_cands"] = g.transform("size")

    rows = []
    for a_id, b_id in zip(c["source1_entity_id"].values, c["candidate_entity_id"].values):
        A, B = s1_rec[a_id], s23_rec[b_id]
        na, nb = A[5], B[5]
        shared = na & nb
        pa = {x for x in na if len(x) in (5, 6)}
        pb = {x for x in nb if len(x) in (5, 6)}
        rows.append((
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
            float(B[9]), float(A[9]), float(B[8]), float(b_id.startswith("S3")),
        ))
    F = pd.DataFrame(rows, columns=FEATURES[5:], index=c.index)
    return pd.concat([c[["source1_entity_id", "candidate_entity_id"] + FEATURES[:5]], F], axis=1)
