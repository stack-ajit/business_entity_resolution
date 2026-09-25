"""
Fast blocking ablation: 10K sample S1 vs (their true matches + ~1.5M random train S2/S3),
in memory. Minutes per variant instead of ~25 min for a full index rebuild. Absolute
recall is optimistic (fewer distractors); use it to RANK variants, then confirm the
winner with evaluate_blocking.py at full scale.

Usage: python src/blocking/ablate_blocking.py ["variant name" ...]
"""
import os, re, sys, time, itertools
import numpy as np, pandas as pd, scipy.sparse as sp
from functools import lru_cache
from anyascii import anyascii
from sklearn.feature_extraction import FeatureHasher
from sparse_dot_topn import sp_matmul_topn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from blocking import tfidf_blocking as tb
from config import SAMPLE_DIR, TRAIN_DIR, CACHE_DIR
T = time.time()
def log(*a): print(f"[{time.time()-T:5.0f}s]", *a, flush=True)

s1 = pd.read_csv(os.path.join(SAMPLE_DIR, "sample_source1.tsv"), sep="\t", dtype=str, keep_default_na=False)
gt = pd.read_csv(os.path.join(SAMPLE_DIR, "sample_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False)
truth = set((a, x) for a, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for x in m.split(",") if x)
os.makedirs(CACHE_DIR, exist_ok=True)
cp = os.path.join(CACHE_DIR, "ablate_corpus.parquet")
if not os.path.exists(cp):
    need = {b for _, b in truth}; rng = np.random.default_rng(0); parts = []
    for i in (2, 3):
        for ch in tb.read_tsv_chunks(os.path.join(TRAIN_DIR, f"train_source{i}.tsv"), 250_000):
            keep = ch.entity_id.isin(need).values | (rng.random(len(ch)) < 0.145)
            parts.append(ch[keep])
    pd.concat(parts).to_parquet(cp)
corpus = pd.read_parquet(cp); log("corpus", len(corpus))

_vow = re.compile(r"[aeiouyh]"); _rep = re.compile(r"(.)\1+")
def skel_v1(t): return _rep.sub(r"\1", _vow.sub("", t))
skel_v1 = lru_cache(None)(skel_v1)
_na = re.compile(r"[^a-z0-9]+")
def norm_v1(x):
    if not x: return []
    return _na.sub(" ", anyascii(x).lower().replace("&", " and ")).split()

def feats(name, addr, o):
    norm = tb.normalize_tokens if o["norm"] == "v2" else norm_v1
    n = [t for t in norm(name) if t not in tb.NAME_STOP]
    a = [tb.ADDR_CANON.get(t, tb.NUMBER_WORDS.get(t, t) if o["norm"] == "v2" else t) for t in norm(addr)]
    nf, af = [], []
    for t in n:
        if len(t) >= 2: nf.append("n_" + t)
        for sk, tag in ((skel_v1, "v1"), (tb.skeleton, "v2")):
            if tag in o["skel"]:
                k = sk(t)
                if len(k) >= 3: nf.append(f"k{tag}_" + k)
        if o["concat"] and len(t) >= 8: nf.append("c_" + t)
    nf += ["b_" + x + "_" + y for x, y in zip(n, n[1:])]
    if o["concat"]:
        for j in (2, 3):
            if len(n) >= j: nf.append("c_" + "".join(n[:j]))
    for i, t in enumerate(a):
        if t.isdigit():
            if o["norm"] == "v2": t = t.lstrip("0") or "0"
            af.append("d_" + t)
            if i + 1 < len(a) and not a[i + 1].isdigit(): af.append("dw_" + t + "_" + a[i + 1])
        elif len(t) >= 3:
            af.append("a_" + t)
            if o["askel"]:
                k = tb.skeleton(t)
                if len(k) >= 3: af.append("ak_" + k)
    return nf, af

H = FeatureHasher(n_features=2**21, input_type="string", alternate_sign=False)
def hsh(lists):
    X = H.transform(lists).tocsr().astype(np.float32); X.data[:] = 1; return X
def mats(df, o):
    fa = [feats(n, a, o) for n, a in zip(df.business_name.values, df.business_address.values)]
    return hsh(f[0] for f in fa), hsh(f[1] for f in fa)
def wn(X, w):
    X = (X @ sp.diags(w)).tocsr(); nr = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); nr[nr == 0] = 1
    return (sp.diags(1 / nr) @ X).astype(np.float32).tocsr()

def run(o, Ks=(10, 20, 30, 50)):
    res = {}
    for c in s1.country.unique():
        q = s1[s1.country == c]; r = corpus[corpus.country == c]
        Qn, Qa = mats(q, o); Xn, Xa = mats(r, o)
        Qc, Xc = sp.hstack([Qn, Qa]).tocsr(), sp.hstack([Xn, Xa]).tocsr()
        df = np.bincount(Xc.indices, minlength=Xc.shape[1])
        # scale df as if corpus were full size (1.5M sample of 10.3M) so min_idf pruning matches production
        idf = np.log(len(r) / (1 + df)).astype(np.float32); idf[df == 0] = 0; idf[idf < np.log(1000)] = 0
        wN, wA = idf[:2**21], idf[2**21:]
        views = {"both": (wn(Qc, idf), wn(Xc, idf))}
        if o["views"]:
            views["name"] = (wn(Qn, wN), wn(Xn, wN)); views["addr"] = (wn(Qa, wA), wn(Xa, wA))
        ids = r.entity_id.values; qids = q.entity_id.values
        for v, (Q, X) in views.items():
            R = sp_matmul_topn(Q, X.T.tocsr(), top_n=max(Ks), n_threads=12, sort=True).tocsr()
            for i in range(len(qids)):
                cols = R.indices[R.indptr[i]:R.indptr[i + 1]]
                res.setdefault(v, {})[qids[i]] = ids[cols]
    out = {}
    for K in Ks:
        if o["views"]:
            # budget split: K total per S1 = K/2 combined + K/4 name + K/4 addr (union, dedup)
            kb, kn = K // 2, K // 4
            pairs = set()
            for qid in res["both"]:
                s = set(res["both"][qid][:kb]) | set(res["name"].get(qid, [])[:kn]) | set(res["addr"].get(qid, [])[:kn])
                pairs |= {(qid, x) for x in s}
        else:
            pairs = {(qid, x) for qid, l in res["both"].items() for x in l[:K]}
        out[K] = (round(len(pairs & truth) / len(truth), 4), len(pairs))
    return out

base = dict(norm="v1", skel=("v1",), concat=False, askel=False, views=False)
variants = {
    "v1 (baseline)": base,
    "+norm v2": {**base, "norm": "v2"},
    "+norm v2, skel v2": {**base, "norm": "v2", "skel": ("v2",)},
    "+norm v2, skel v1+v2": {**base, "norm": "v2", "skel": ("v1", "v2")},
    "+norm v2, concat": {**base, "norm": "v2", "concat": True},
    "+norm v2, askel": {**base, "norm": "v2", "askel": True},
    "v2 (all)": {**base, "norm": "v2", "skel": ("v2",), "concat": True, "askel": True},
    "final (norm v2, skel v2, askel)": {**base, "norm": "v2", "skel": ("v2",), "askel": True},
    "v1 + views": {**base, "views": True},
    "+norm v2 + views": {**base, "norm": "v2", "views": True},
}
only = sys.argv[1:] 
for name, o in variants.items():
    if only and name not in only: continue
    log(f"{name:28s}", run(o))
