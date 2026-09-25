"""
Memory-efficient candidate generation (blocking) via hashed TF-IDF top-K retrieval.

Why not exact-string keys: true pairs rarely agree on the *first* address word
(addresses get reordered) and non-Latin names were being wiped to "". But 98% of
true pairs share at least one name token or one address number. So every record
becomes a bag of IDF-weighted features and each Source 1 record retrieves its
top-K most similar Source 2/3 records (OR semantics, rare tokens dominate).

Memory: Source 2/3 is streamed in chunks. Pass 1 hashes features (no vocabulary
kept in RAM), caches each chunk's sparse matrix to disk and accumulates document
frequencies per country. Pass 2 scores query batches against each cached chunk
and merges a running top-K. Peak RAM is bounded by chunk size, not corpus size.
"""
import os
import re
import glob
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anyascii import anyascii
from sklearn.feature_extraction import FeatureHasher
from sparse_dot_topn import sp_matmul_topn

N_FEATURES = 2 ** 22
_hasher = FeatureHasher(n_features=N_FEATURES, input_type="string", alternate_sign=False)
_non_alnum = re.compile(r"[^a-z0-9]+")
_zero_in_word = re.compile(r"(?<=[a-z])0(?=[a-z])")       # "f0ods" -> "foods"
_ordinal = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")         # "26th"/"158nd" -> "26"/"158"
_digit_alpha = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")
_repeats = re.compile(r"(.)\1+")
# Phonetic folding, applied in order. Chosen so that anyascii transliterations of
# Indic scripts collide with the English spelling ("phainyans"~"finance",
# "praibhet"~"private", "mainejmemt"~"management", "imdiyn"~"indian").
_PHONETIC = [(re.compile(p), r) for p, r in [
    (r"tion", "sn"), (r"gh", ""), (r"ng\b", "n"), (r"ph", "f"), (r"bh", "b"),
    (r"kh", "k"), (r"dh", "d"), (r"th", "t"), (r"sh", "s"), (r"ck", "k"),
    (r"c(?=[eiy])", "s"), (r"[cq]", "k"), (r"x", "ks"), (r"[vw]", "b"),
    (r"g", "j"), (r"z", "j"), (r"m", "n"), (r"[aeiouyh]", ""),
]]
NUMBER_WORDS = {w: str(i) for i, ws in enumerate([
    (), ("one", "first"), ("two", "second"), ("three", "third"), ("four", "fourth"),
    ("five", "fifth"), ("six", "sixth"), ("seven", "seventh"), ("eight", "eighth"),
    ("nine", "ninth"), ("ten", "tenth")]) for w in ws}

# Only tokens that carry no identity. Everything else (incl. France's "sarl"/"rue")
# is handled by IDF, which is learned from the data, so no country is hard-coded.
NAME_STOP = {"the", "and", "of", "dba", "formerly"}
ADDR_CANON = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd", "drive": "dr",
    "lane": "ln", "court": "ct", "place": "pl", "highway": "hwy", "suite": "ste",
    "apartment": "apt", "floor": "fl", "near": "nr", "opposite": "opp",
}


def normalize_tokens(text):
    """Transliterate any script to ASCII, lowercase, split on non-alphanumerics."""
    if not isinstance(text, str) or not text:
        return []
    if not text.isascii():
        text = anyascii(text)
    s = _non_alnum.sub(" ", text.lower().replace("&", " and "))
    s = _ordinal.sub(r"\1", _zero_in_word.sub("o", s))
    return _digit_alpha.sub(" ", s).split()


@lru_cache(maxsize=2_000_000)
def skeleton(token):
    """Phonetic consonant skeleton ("marketing" -> "nrktn"): robust to vowel typos and
    to transliteration drift between Indic scripts and English spellings."""
    for pat, rep in _PHONETIC:
        token = pat.sub(rep, token)
    return _repeats.sub(r"\1", token)


def record_features(name, address):
    """Bag of prefixed string features for one record."""
    n = [t for t in normalize_tokens(name) if t not in NAME_STOP]
    a = [ADDR_CANON.get(t, NUMBER_WORDS.get(t, t)) for t in normalize_tokens(address)]
    feats = []
    for t in n:
        if len(t) >= 2:
            feats.append("n_" + t)
        k = skeleton(t)
        if len(k) >= 3:
            feats.append("k_" + k)
        if len(t) >= 8:  # may be a run-together name: "bergersociety", "fooltd.com"
            feats.append("c_" + t)
    feats += ["b_" + x + "_" + y for x, y in zip(n, n[1:])]
    # run-together prefixes match domain/hashtag names ("capitalreliablenetworks.com")
    for j in (2, 3):
        if len(n) >= j:
            feats.append("c_" + "".join(n[:j]))
    for i, t in enumerate(a):
        if t.isdigit():
            t = t.lstrip("0") or "0"  # "002050" == "2050"
            feats.append("d_" + t)
            if i + 1 < len(a) and not a[i + 1].isdigit():
                feats.append("dw_" + t + "_" + a[i + 1])
        elif len(t) >= 3:
            feats.append("a_" + t)
            k = skeleton(t)
            if len(k) >= 3:
                feats.append("ak_" + k)  # address typos: "morgantno" ~ "morganton"
    return feats


def hash_records(names, addresses):
    """Binary hashed feature matrix (CSR, float32) for parallel name/address arrays."""
    X = _hasher.transform(record_features(n, a) for n, a in zip(names, addresses))
    X = X.tocsr().astype(np.float32)
    X.data[:] = 1.0
    return X


def read_tsv_chunks(path, chunksize=500_000):
    """Stream a challenge TSV as string chunks (empty string for missing fields)."""
    return pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str,
                       keep_default_na=False, na_filter=False)


def build_index(paths, cache_dir, chunksize=250_000, shard_rows=2_000_000, min_idf=None, log=print):
    """
    Pass 1: hash every Source 2/3 record into on-disk chunks and count per-country DF.
    Pass 1b: regroup rows per country into shards of <= shard_rows, already IDF-weighted,
    L2-normalized and transposed, so queries only load + multiply (no per-batch rework).
    min_idf: features with IDF below this (default log(1000), i.e. present in >0.1% of a
    country's records) are dropped from retrieval - they barely move cosine but blow up
    matmul cost. They remain available to the downstream matching model.
    """
    os.makedirs(cache_dir, exist_ok=True)
    min_idf = np.log(1000.0) if min_idf is None else min_idf
    df_by_country, n_by_country = {}, {}
    k = 0
    for path in paths:
        for chunk in read_tsv_chunks(path, chunksize):
            X = hash_records(chunk["business_name"].values, chunk["business_address"].values)
            country = chunk["country"].values.astype(str)
            sp.save_npz(os.path.join(cache_dir, f"chunk{k:03d}_X.npz"), X, compressed=False)
            np.save(os.path.join(cache_dir, f"chunk{k:03d}_ids.npy"), chunk["entity_id"].values.astype(str))
            np.save(os.path.join(cache_dir, f"chunk{k:03d}_country.npy"), country)
            for c in np.unique(country):
                rows = X[country == c]
                df = df_by_country.setdefault(c, np.zeros(N_FEATURES, np.int64))
                df += np.bincount(rows.indices, minlength=N_FEATURES)
                n_by_country[c] = n_by_country.get(c, 0) + rows.shape[0]
            log(f"hashed chunk {k} from {os.path.basename(path)}: {X.shape[0]} rows")
            k += 1
    idf = {}
    for c, df in df_by_country.items():
        idf[c] = np.log(n_by_country[c] / (1.0 + df)).astype(np.float32)
        idf[c][df == 0] = 0.0
        idf[c][idf[c] < min_idf] = 0.0
    np.savez(os.path.join(cache_dir, "idf.npz"), **{f"c_{c}": v for c, v in idf.items()})

    chunk_stems = [f[:-len("_X.npz")] for f in sorted(glob.glob(os.path.join(cache_dir, "chunk*_X.npz")))]
    for ci, c in enumerate(sorted(idf)):
        parts, part_ids, n_rows, shard = [], [], 0, 0

        def flush():
            nonlocal parts, part_ids, n_rows, shard
            if not parts:
                return
            XT = _weight_and_normalize(sp.vstack(parts, format="csr"), idf[c]).T.tocsr()
            sp.save_npz(os.path.join(cache_dir, f"shard_c{ci}_{shard:02d}_XT.npz"), XT, compressed=False)
            np.save(os.path.join(cache_dir, f"shard_c{ci}_{shard:02d}_ids.npy"), np.concatenate(part_ids))
            log(f"country {c}: shard {shard} with {XT.shape[1]} rows")
            parts, part_ids, n_rows, shard = [], [], 0, shard + 1

        for stem in chunk_stems:
            rows = np.flatnonzero(np.load(stem + "_country.npy") == c)
            if len(rows):
                parts.append(sp.load_npz(stem + "_X.npz")[rows])
                part_ids.append(np.load(stem + "_ids.npy")[rows])
                n_rows += len(rows)
            if n_rows >= shard_rows:
                flush()
        flush()
    with open(os.path.join(cache_dir, "countries.json"), "w", encoding="utf-8") as f:
        json.dump({c: ci for ci, c in enumerate(sorted(idf))}, f)
    for stem in chunk_stems:  # raw chunks are no longer needed
        for suffix in ("_X.npz", "_ids.npy", "_country.npy"):
            os.remove(stem + suffix)
    return idf


def load_idf(cache_dir):
    z = np.load(os.path.join(cache_dir, "idf.npz"))
    return {k[2:]: z[k] for k in z.files}


def _weight_and_normalize(X, w):
    """Apply (already min_idf-pruned) IDF weights and L2-normalize rows."""
    X = X @ sp.diags(w, format="csr")
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sp.diags(1.0 / norms, format="csr") @ X).astype(np.float32).tocsr()


def query_index(q_ids, q_names, q_addresses, q_countries, cache_dir, top_k=50,
                batch_size=200_000, n_threads=None, log=print):
    """
    Pass 2: top-K retrieval for Source 1 records against the sharded index.

    Returns a DataFrame (source1_entity_id, candidate_entity_id, score), score = cosine.
    Queries only compare against Source 2/3 rows with the same country label; a
    country unseen in the index simply yields no candidates.
    """
    n_threads = n_threads or os.cpu_count()
    idf = load_idf(cache_dir)
    with open(os.path.join(cache_dir, "countries.json"), encoding="utf-8") as f:
        cidx = json.load(f)
    q_ids = np.asarray(q_ids)
    q_countries = np.asarray(q_countries).astype(str)
    out = []
    for c in np.unique(q_countries):
        if c not in idf:
            log(f"country {c!r} not present in index, no candidates")
            continue
        shards = sorted(glob.glob(os.path.join(cache_dir, f"shard_c{cidx[c]}_*_XT.npz")))
        sel = np.flatnonzero(q_countries == c)
        for b0 in range(0, len(sel), batch_size):
            t0 = time.time()
            bi = sel[b0:b0 + batch_size]
            Q = _weight_and_normalize(hash_records(q_names[bi], q_addresses[bi]), idf[c])
            best_s = np.full((len(bi), top_k), -1.0, np.float32)
            best_id = np.full((len(bi), top_k), "", dtype=object)
            for f in shards:
                R = sp_matmul_topn(Q, sp.load_npz(f), top_n=top_k, n_threads=n_threads).tocsr()
                ids = np.load(f[:-len("_XT.npz")] + "_ids.npy", allow_pickle=False)
                # scatter this shard's top-K into a dense (n, top_k) block, then merge
                cs = np.full((len(bi), top_k), -1.0, np.float32)
                cid = np.full((len(bi), top_k), "", dtype=object)
                counts = np.diff(R.indptr)
                r_idx = np.repeat(np.arange(len(bi)), counts)
                pos = np.arange(R.nnz) - np.repeat(R.indptr[:-1], counts)
                cs[r_idx, pos] = R.data
                cid[r_idx, pos] = ids[R.indices]
                if len(shards) == 1:
                    best_s, best_id = cs, cid
                    break
                all_s = np.hstack([best_s, cs])
                all_id = np.hstack([best_id, cid])
                keep = np.argpartition(-all_s, top_k - 1, axis=1)[:, :top_k]
                best_s = np.take_along_axis(all_s, keep, 1)
                best_id = np.take_along_axis(all_id, keep, 1)
            ok = best_s > 0
            rr = np.nonzero(ok)
            out.append(pd.DataFrame({
                "source1_entity_id": q_ids[bi][rr[0]],
                "candidate_entity_id": best_id[ok],
                "score": best_s[ok],
            }))
            log(f"country {c}: queries {b0}-{b0 + len(bi)} of {len(sel)} done in {time.time() - t0:.0f}s")
    if not out:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])
    return pd.concat(out, ignore_index=True)
