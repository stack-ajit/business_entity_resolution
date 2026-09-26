"""
Pair features for the matching model: given candidate pairs (S1 record, S2/S3 record)
from blocking, describe how similar the two records are.

Two feature tiers (two-stage model):
- CHEAP_FEATURES: retrieval context only (forward score/rank + reverse "which S1 does this
  candidate itself point to"). Vectorised, no string work -> used by the stage-1 pruner
  that shrinks the candidate set.
- FEATURES: cheap + string similarities + IDF-weighted token overlaps -> stage-2 matcher,
  computed only for the pairs that survive stage 1.

Design notes:
- Country is deliberately NOT a feature: the test set contains France, unseen in
  training. IDF weights are per-country (learned from each country's own records), so
  "common word" is judged within the right language automatically.
- Reverse features encode the one-to-one structure of the data: each S2/S3 record
  belongs to at most one S1 entity.
"""
import itertools
import multiprocessing as mp
import os
import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.utils import murmurhash3_32

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from blocking.tfidf_blocking import (ADDR_CANON, N_FEATURES, NAME_STOP, NUMBER_WORDS,
                                     normalize_tokens, read_tsv_chunks, skeleton)
from blocking.reverse import id_code

# Legal / generic suffixes: compared separately so "X Pvt Ltd" vs "X" is not penalised
# on the core name. Includes France forms; unknown suffixes just stay in the core name.
LEGAL = {
    "pvt", "private", "ltd", "limited", "llc", "inc", "incorporated", "co", "company",
    "corp", "corporation", "llp", "pllc", "lp", "plc", "pc", "pa", "the", "and", "of",
    "sas", "sarl", "sa", "sasu", "eurl", "sci", "com", "www", "dba", "formerly",
}
_digits = re.compile(r"\d+")
REV_TOP_R = 3  # reverse list length (see blocking/reverse.py)


@lru_cache(maxsize=4_000_000)
def _hidx(feat):
    """Hash bucket of a blocking feature string (identical to FeatureHasher's)."""
    return abs(murmurhash3_32(feat, seed=0)) % N_FEATURES


def _weights(name_tokens, addr_tokens, idf):
    """IDF weight per name / address token, using the blocking features' names."""
    if idf is None:
        return {}, {}
    nw = {t: float(idf[_hidx("n_" + t)]) for t in name_tokens if t not in NAME_STOP and len(t) >= 2}
    aw = {}
    for t in addr_tokens:
        t = ADDR_CANON.get(t, NUMBER_WORDS.get(t, t))
        if t.isdigit():
            f = "d_" + (t.lstrip("0") or "0")
        elif len(t) >= 3:
            f = "a_" + t
        else:
            continue
        aw[f] = float(idf[_hidx(f)])
    return nw, aw


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_ORD = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
        "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
        "seventeenth", "eighteenth", "nineteenth"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90, "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
         "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90}
_WORDNUM = {**{w: i for i, w in enumerate(_ONES)}, **{w: i for i, w in enumerate(_ORD)}}


def _numbers(tokens):
    """Ordered integer values in an address: digits plus number/ordinal words up to 99
    ('fourteenth' -> 14, 'twenty first' -> 21). Leading zeros vanish ('017954' -> 17954)."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t.isdigit() and len(t) <= 12:
            out.append(int(t))
        elif t in _TENS:
            v = _TENS[t]
            if i + 1 < len(tokens) and tokens[i + 1] in _WORDNUM and _WORDNUM[tokens[i + 1]] < 10:
                v += _WORDNUM[tokens[i + 1]]
                i += 1
            out.append(v)
        elif t in _WORDNUM and t not in ("zero", "zeroth", "one", "first", "second"):
            # 'one'/'first'/'second' are too often plain words ("first floor") to count
            out.append(_WORDNUM[t])
        i += 1
    return tuple(out)


def prepare(name, address, idf=None):
    """Prepared tuple used by pair_features (normalized strings, sets, flags, IDF weights,
    ordered address numbers, name initials)."""
    nt = normalize_tokens(name)
    core = [t for t in nt if t not in LEGAL]
    at = normalize_tokens(address)
    nums = {t.lstrip("0") or "0" for t in at if t.isdigit()}
    nw, aw = _weights(nt, at, idf)
    initials = "".join(t[0] for t in core) if len(core) >= 2 else ""
    return (" ".join(nt), " ".join(core), " ".join(skeleton(t) for t in core),
            "".join(core), " ".join(at), nums, set(core), set(at),
            bool(name) and not name.isascii(), address == "", nw, aw,
            _numbers(at), initials)


def load_records(paths, ids=None, idf_by_country=None):
    """Load records (optionally only the given entity ids) as {id: prepared tuple}."""
    ids = None if ids is None else set(ids)
    idf_by_country = idf_by_country or {}
    out = {}
    for p in paths:
        for ch in read_tsv_chunks(p, 500_000):
            if ids is not None:
                ch = ch[ch["entity_id"].isin(ids)]
            for i, n, a, c in zip(ch["entity_id"].values, ch["business_name"].values,
                                  ch["business_address"].values, ch["country"].values):
                out[i] = prepare(n, a, idf_by_country.get(c))
    return out


def load_raw(paths, ids=None):
    """{id: (name, address, country)} (optionally only given ids) - prepare lazily per batch."""
    ids = None if ids is None else set(ids)
    out = {}
    for p in paths:
        for ch in read_tsv_chunks(p, 500_000):
            if ids is not None:
                ch = ch[ch["entity_id"].isin(ids)]
            out.update(zip(ch["entity_id"].values,
                           zip(ch["business_name"].values, ch["business_address"].values,
                               ch["country"].values)))
    return out


def _jacc(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _wsim(wa, wb):
    """IDF-weighted overlap: jaccard, coverage of A, coverage of B, heaviest unmatched token each side."""
    sa, sb = sum(wa.values()), sum(wb.values())
    sh = sum(w for t, w in wa.items() if t in wb)
    un = sa + sb - sh
    return (sh / un if un > 0 else 0.0, sh / sa if sa > 0 else 0.0, sh / sb if sb > 0 else 0.0,
            max((w for t, w in wa.items() if t not in wb), default=0.0),
            max((w for t, w in wb.items() if t not in wa), default=0.0))


CHEAP_FEATURES = [
    "score", "rank", "score_rel", "score_gap_best", "fwd",
    "rscore", "rev_rank", "rev_score_rel", "rev_gap", "rev_margin", "rev_is_best", "rev_best",
    "a_n_revbest", "is_anchor", "sib_n", "sib_best",
]
STRING_FEATURES = [
    "name_ratio", "name_tset", "name_tsort", "name_partial", "name_jw",
    "core_ratio", "core_tset", "core_partial", "core_concat_ratio", "skel_ratio", "skel_tset",
    "core_jacc", "name_len_a", "name_len_b",
    "addr_ratio", "addr_tset", "addr_partial", "addr_jacc",
    "num_jacc", "num_shared", "num_conflict", "num_a", "num_b", "max_shared_num_len",
    "postcode_match", "postcode_conflict",
    "addr_empty_b", "addr_empty_a", "name_nonascii_b", "is_s3",
    "name_w_jacc", "name_w_cov_a", "name_w_cov_b", "name_w_miss_a", "name_w_miss_b",
    "addr_w_jacc", "addr_w_cov_a", "addr_w_cov_b", "addr_w_miss_a", "addr_w_miss_b",
    # generator-aware: decoys are near-copies with a nudged house number (428 -> 435) or a
    # swapped name word; true matches get truncation/padding (1082 -> 108) and acronyms (HT)
    "num1_eq", "num1_diff", "num1_prefix", "num_near_miss", "num_min_diff",
    "acro_ab", "acro_ba", "name_extra_a", "name_extra_b",
]


def _numfeat(la, lb):
    """First-number relation + near-miss counts between two ordered number tuples."""
    if not la or not lb:
        return (-1.0, -1.0, 0.0, 0.0, -1.0)
    a1, b1 = la[0], lb[0]
    sa, sb = str(a1), str(b1)
    prefix = float(a1 != b1 and (sa.startswith(sb) or sb.startswith(sa)))
    setb = set(lb)
    near, mind = 0, None
    for x in la:
        if x in setb:
            continue
        d = min(abs(x - y) for y in lb)
        mind = d if mind is None else min(mind, d)
        near += 0 < d <= 20
    return (float(a1 == b1), float(min(abs(a1 - b1), 1_000_000)), prefix, float(near),
            -1.0 if mind is None else float(min(mind, 1_000_000)))
FEATURES = CHEAP_FEATURES + STRING_FEATURES


def union_reverse_candidates(cands, rev, s1_ids):
    """
    Add pairs found only from the S2/S3 side: (S1, B) where S1 is among B's top-R S1 matches
    but B was not in S1's forward top-K. Marked fwd=0, forward score 0.
    """
    from blocking.reverse import decode_s1, decode_s23
    c = cands.assign(fwd=1)
    codes = id_code(s1_ids)
    r = rev[rev["s1"].isin(codes)]
    extra = pd.DataFrame({"source1_entity_id": decode_s1(r["s1"].values),
                          "candidate_entity_id": decode_s23(r["b"].values),
                          "score": np.float32(0.0), "fwd": 0})
    c = pd.concat([c, extra], ignore_index=True)
    return c.drop_duplicates(["source1_entity_id", "candidate_entity_id"], keep="first").reset_index(drop=True)


def sibling_expand(cands, rev, raw, s23_index, n_anchor=5, top_n=5):
    """
    Sibling expansion: an entity's true matches are noisy copies of the same business, so
    they resemble each other. For each S1, take up to n_anchor "anchor" candidates (forward
    top-3 plus candidates whose own best S1 is this one), look up each anchor's top_n most
    similar S2/S3 records, and
      - add those neighbours as new candidates (recovers matches retrieval ranked too low;
        measured on the sample: 55% of blocking misses are a found sibling's top-5 neighbour)
      - give every pair sib_n (how many of this S1's anchors list it as a neighbour) and
        sib_best (strongest such similarity), plus is_anchor.
    raw: {s23_id: (name, address, country)} for at least the anchors.
    """
    from blocking.tfidf_blocking import query_index
    c = cands.copy()
    if "fwd" not in c:
        c["fwd"] = 1
    c["_r"] = c.groupby("source1_entity_id")["score"].rank(ascending=False, method="first")
    c["_a"] = id_code(c["source1_entity_id"].values)
    c["_b"] = id_code(c["candidate_entity_id"].values)
    top1 = rev.loc[(rev["rrank"] == 1) & rev["b"].isin(c["_b"].unique()), ["b", "s1"]]
    c = c.merge(top1.rename(columns={"b": "_b", "s1": "_a"}).assign(_own=True), on=["_b", "_a"], how="left")
    c["_own"] = c["_own"].fillna(False).astype(bool)
    c = c.drop(columns=["_a", "_b"])
    anc = c[(c["_r"] <= 3) | c["_own"]].sort_values(["source1_entity_id", "_own", "score"],
                                                     ascending=[True, False, False])
    anc = anc[anc.groupby("source1_entity_id").cumcount() < n_anchor][["source1_entity_id", "candidate_entity_id"]]
    c["is_anchor"] = 0.0
    c.loc[c.set_index(["source1_entity_id", "candidate_entity_id"]).index.isin(
        anc.set_index(["source1_entity_id", "candidate_entity_id"]).index), "is_anchor"] = 1.0

    a_ids = anc["candidate_entity_id"].unique()
    recs = [raw[i] for i in a_ids]
    nb = query_index(a_ids, np.array([r[0] for r in recs], dtype=object),
                     np.array([r[1] for r in recs], dtype=object),
                     np.array([r[2] for r in recs], dtype=object), s23_index,
                     top_k=top_n + 1, log=lambda m: None)
    nb = nb[nb["source1_entity_id"] != nb["candidate_entity_id"]]  # drop self-match
    nb = nb.rename(columns={"source1_entity_id": "anchor", "candidate_entity_id": "candidate_entity_id",
                            "score": "nscore"})
    sib = anc.rename(columns={"candidate_entity_id": "anchor"}).merge(nb, on="anchor")
    sib = sib.groupby(["source1_entity_id", "candidate_entity_id"]).agg(
        sib_n=("anchor", "nunique"), sib_best=("nscore", "max")).reset_index()

    c = c.merge(sib, on=["source1_entity_id", "candidate_entity_id"], how="outer")
    new = c["score"].isna()
    c.loc[new, ["score", "fwd", "is_anchor"]] = 0.0
    c["sib_n"] = c["sib_n"].fillna(0).astype(np.float32)
    c["sib_best"] = c["sib_best"].fillna(0).astype(np.float32)
    c["score"] = c["score"].astype(np.float32)
    return c.drop(columns=["_r", "_own"]).reset_index(drop=True)


def add_context_features(cands, rev, rev_sum):
    """Forward retrieval context + reverse (one-to-one) features. Vectorised."""
    c = cands.copy()
    if "fwd" not in c:
        c["fwd"] = 1
    g = c.groupby("source1_entity_id")["score"]
    c["rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    best = g.transform("max")
    c["score_rel"] = (c["score"] / best.where(best > 0)).fillna(0.0).astype(np.float32)
    c["score_gap_best"] = (best - c["score"]).astype(np.float32)

    c["a_code"] = id_code(c["source1_entity_id"].values)
    c["b_code"] = id_code(c["candidate_entity_id"].values)
    r = rev[rev["b"].isin(c["b_code"].unique())].rename(columns={"b": "b_code", "s1": "a_code"})
    c = c.merge(r[["b_code", "a_code", "rscore", "rrank"]], on=["b_code", "a_code"], how="left")
    c = c.merge(rev_sum, left_on="b_code", right_index=True, how="left")
    c["rscore"] = c["rscore"].fillna(0.0).astype(np.float32)
    c["rev_rank"] = c["rrank"].fillna(REV_TOP_R + 1).astype(np.float32)
    c["rev_best"] = c["rev_best"].fillna(0.0).astype(np.float32)
    c["rev_second"] = c["rev_second"].fillna(0.0).astype(np.float32)
    c["rev_score_rel"] = (c["rscore"] / c["rev_best"].where(c["rev_best"] > 0)).fillna(0.0).astype(np.float32)
    c["rev_gap"] = (c["rev_best"] - c["rscore"]).astype(np.float32)
    c["rev_margin"] = (c["rev_best"] - c["rev_second"]).astype(np.float32)
    c["rev_is_best"] = (c["rev_rank"] == 1).astype(np.float32)
    # how many of this S1's candidates point back to it as their best S1 (entity cardinality hint)
    c["a_n_revbest"] = c.groupby("source1_entity_id")["rev_is_best"].transform("sum").astype(np.float32)
    return c.drop(columns=["rrank", "rev_second", "a_code", "b_code"])


def _rows(payload):
    """
    String features for a chunk of pairs. payload = list of (A, B, is_s3) where A/B are
    prepared record tuples. Returns a compact float32 array (n, n_string_features): a list
    of Python float tuples would cost ~1 KB per pair (~7 GB for 7.5M pairs).
    """
    out = np.empty((len(payload), len(STRING_FEATURES)), dtype=np.float32)
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
            *_wsim(A[10], B[10]), *_wsim(A[11], B[11]),
            *_numfeat(A[12], B[12]),
            float(bool(A[13]) and (A[13] in B[6] or A[13] == B[3])),
            float(bool(B[13]) and (B[13] in A[6] or B[13] == A[3])),
            float(len(A[6] - B[6])), float(len(B[6] - A[6])),
        )
    return out


def _payloads(a_ids, b_ids, s1_rec, s23_rec, chunk):
    for i in range(0, len(a_ids), chunk):
        yield [(s1_rec[a], s23_rec[b], b.startswith("S3"))
               for a, b in zip(a_ids[i:i + chunk], b_ids[i:i + chunk])]


def pair_features(ctx, s1_rec, s23_rec, n_jobs=None, chunk=20_000):
    """
    ctx: output of add_context_features. Appends STRING_FEATURES.
    n_jobs: worker processes for the string similarities (default: all cores). Workers get
    only the small chunk of record tuples they score (never the big dicts), so memory stays
    flat; results come back as float32 arrays. Falls back to 1 process on any pool error.
    """
    a_ids = ctx["source1_entity_id"].values
    b_ids = ctx["candidate_entity_id"].values
    n_jobs = n_jobs or os.cpu_count() or 1
    parts = None
    if n_jobs > 1 and len(ctx) > 4 * chunk:
        try:
            parts = []
            gen = _payloads(a_ids, b_ids, s1_rec, s23_rec, chunk)
            # spawn, not fork: a forked worker inherits the parent's multi-GB record dicts and
            # Python's GC touching them copies those pages into every worker (OOM on Kaggle)
            with mp.get_context("spawn").Pool(n_jobs) as pool:
                # bounded waves: Pool.imap would drain the generator into its queue at once
                while wave := list(itertools.islice(gen, n_jobs * 2)):
                    parts += pool.map(_rows, wave, chunksize=1)
        except Exception as e:  # never lose a long run to a pool problem
            print(f"parallel features failed ({e!r}), falling back to 1 process", flush=True)
            parts = None
    if parts is None:
        parts = [_rows(p) for p in _payloads(a_ids, b_ids, s1_rec, s23_rec, chunk)]
    X = np.vstack(parts) if parts else np.empty((0, len(STRING_FEATURES)), np.float32)
    F = pd.DataFrame(X, columns=STRING_FEATURES, index=ctx.index)
    return pd.concat([ctx, F], axis=1)
