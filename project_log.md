# Amazon ML Challenge 2026: Project Development Log

*This document serves as a live tracker of our logic, decisions, and implementation progress throughout the 72-hour hackathon.*

---

## 1. Project Initialization & Strategy Formulation
**Status:** Completed
**Logic & Decisions:**
- Adopted a problem-agnostic repository structure optimized for rapid experimentation.
- Task is Entity Resolution relying strictly on `business_name` and `business_address`.
- **Evaluation Metric:** Macro $F_{0.5}$ score. ML models must prioritize Precision.

---

## 2. Phase 1: Data Subsampling
**Status:** Completed
**Logic & Decisions:**
- Subsampled 10,000 Source 1 entities and generated a dataset representing true matches and random noise from Source 2/3.
- Avoids OOM errors and slow iteration times.

---

## 3. Phase 2: EDA & Text Preprocessing
**Status:** Completed
**Logic & Decisions:**
- Implemented `src/cleaning/text_cleaner.py` to handle lowercasing, non-alphanumeric removal, and standardizing common abbreviations (e.g., "st" to "street").

---

## 4. Phase 3: Candidate Generation (Blocking)
**Status:** In Progress (Needs Refinement)
**Logic & Decisions:**
- **Attempt 1 (TF-IDF KNN):** Attempted a robust math-based approach. Failed due to `MemoryError` allocating the dense matrix on local hardware.
- **Attempt 2 (Multi-Key Pandas Merge):** Generated 5 overlapping text keys (e.g., "first 4 chars name + first word address"). 
- **Results:** 
  - Candidates Generated: ~844,000
  - Blocking Recall: 53.42%
- **Decision:** A 53% recall is too low for the final submission, as it limits our maximum possible score. The data is extremely noisy (typos/missing words break strict string keys). 
- **Next Action:** We will accept the 844,000 candidate pairs for now so we can move forward and build the **Feature Engineering** and **ML Model** pipelines. Once the full pipeline works end-to-end, we will return to Phase 3 and implement an advanced, memory-efficient blocking strategy like `MinHash LSH` or chunked TF-IDF.

---

### Phase 3 — Attempt 3: Hashed TF-IDF Top-K Retrieval (replaces multi-key merge)
**Status:** Implemented, full-scale validation running
**Code:** `src/blocking/tfidf_blocking.py`, `src/blocking/evaluate_blocking.py`

#### Why the 5-key merge capped at 53% (error analysis on all 34,567 true pairs in the sample)
| Root cause | Evidence | Effect on old keys |
|---|---|---|
| Address components are reordered (`"AL, Talladega, 203 26th Street"` vs `"203-B 26rd St, Talladega, Alabama"`) | Only **52.9%** of true pairs share the first address word | 4 of 5 keys use `addr_first_word` → mostly dead |
| Keys are conjunctive (name-part AND address-part) | Both fields must survive noise at the same time | A single typo/noise prefix (`">> Metropolitan"`, `"LLC Moncada"`, `"Pvt. EFS..."`) breaks a key |
| Non-Latin scripts erased by `[^a-z0-9]` | 13.7% of matched S2/S3 names are Devanagari/Kannada/etc.; 6.4% of names become `""` | Recall on those pairs: **28.7%** |
| Missing addresses | 4.4% of matched records have no address | Recall on them: **0%** (every key needs the address) |
| Per-country / per-source | India 43.1% vs US 60.1%; S3 47.2% vs S2 60.0% | India + S3 are noisier |

**Key finding:** 97.9% of true pairs share *at least one* name token OR one address number. The information needed for high recall exists; exact-string AND-keys throw it away.

Another finding: only **5.6% of S1 entities are singletons** (mean 3.5 matches, max 10). So even though F0.5 favours precision, recall still matters for ~94% of entities, and blocking recall caps the score directly.

#### What changed and why
1. **OR-semantics instead of AND-keys.** Each record becomes a bag of features, and each S1 record retrieves the **top-K most similar** S2/S3 records by IDF-weighted cosine. One strong shared feature (a rare name word, or house number + street) is enough to surface a match. The old approach needed an exact key collision.
2. **Transliteration (`anyascii`, ISC license) before cleaning.** `राम मार्केटिंग प्राइवेट लिमिटेड` → `ram marketimg praivet limited` instead of `""`. This recovers the non-Latin names.
3. **Features per record:**
   - `n_` name word tokens; `b_` name word bigrams (word order / specificity)
   - `k_` consonant skeleton of each name word (`marketing`→`mrktng`), robust to vowel typos and transliteration drift (`Ambience`/`Amcbiec`, `Ratnakar`/`Ratnakra`)
   - `d_` address numbers (house/plot/PIN/ZIP), `dw_` number+next word (`203_26th`); very specific
   - `a_` address words, with street-type canonicalisation (`street`→`st`, `road`→`rd`)
   - Order-independent, so address reordering no longer matters. A record with no address can still match on name.
4. **IDF learned per country instead of hard-coded suffix lists.** Words like `pvt`, `llc`, `road`, and for test **France** `sarl`, `rue`, get low weight automatically. Nothing is hard-coded to {US, India}, as the rules require. Features present in >0.1% of a country's records are dropped from retrieval, which bounds compute.
5. **Blocking within the same country label.** Every true pair in the sample has the same country on both sides.
6. **Memory-safe streaming (fixes the Attempt-1 `MemoryError`).** Dense TF-IDF / KNN at 1.7M × 10M is impossible on a 16 GB machine.
   - `FeatureHasher` (2^22 buckets): no vocabulary in RAM.
   - Pass 1 streams S2/S3 in 250K-row chunks, caches each sparse chunk to `data/cache/train_index/` and accumulates per-country document frequencies.
   - Pass 2 scores S1 query batches against each chunk with `sparse_dot_topn` (multithreaded sparse matmul + top-K, Apache-2.0) and merges a running top-K.
   - Peak RAM depends on chunk size, not corpus size.
7. **Honest evaluation.** The old recall used a 134K-record noise pool, which flatters any method. `evaluate_blocking.py` queries the 10K sampled S1 against the **full 10.3M** train S2/S3 and reports recall@K, recall at adaptive score cut-offs, per-country/per-source recall, and dumps missed pairs to `data/cache/missed_pairs.tsv` for error analysis.

#### Results
| Setup | Recall | Candidates |
|---|---|---|
| Attempt 2: 5 keys, sample noise pool | 53.42% | ~844K |
| Attempt 3: top-20, sample noise pool (smoke test, 2K queries) | **97.51%** | 38K (≤20 per S1) |
| Attempt 3 (v1): full 10.3M train S2/S3, K=20 | **90.55%** | 200K (20 per S1) |
| Attempt 3 (v1): full 10.3M, K=50 | 93.20% | 500K |
| Attempt 3 (v1): full 10.3M, K=100 | 94.35% | 1.0M |

v1 full-scale detail: recall@5 77.9%, @10 86.9%, @30 92.0%, @75 94.0%. Relative cut-off `score ≥ 0.3·best` gives 92.4% with ~39 per S1. At K=100: India 92.2% vs US 95.8%; S2 94.5% ≈ S3 94.3%. For comparison, the old 5-key method got 53% against the *easier* 134K pool, so its full-scale recall would be lower still.

### Phase 3 — Attempt 3 v2: fixes from error analysis of the 1,953 v1 misses
In the missed pairs, 29.6% have a non-Latin S2/S3 name and 8.7% have an empty address. Patterns found, and the fix for each:

| Miss pattern (real examples) | Fix | Why it works |
|---|---|---|
| `তিরুপতি ফাইন্যান্স` → `tirupti phainyans` vs `Tirupati Finance`; `mainejmemt` vs `management`; `imdiyn` vs `indian` | **Phonetic skeleton**: ph→f, bh/v/w→b, kh→k, th→t, g/z→j, m→n, silent `gh`, `tion`→`sn`, `c`→s/k, drop vowels, collapse repeats | Transliteration and English spelling differ in vowels, aspiration and nasals, not consonant order. After folding, all 7 checked Indic misses produce **identical** keys (`fns`=`fns`, `njnt`=`njnt`, `knstrksn`=`knstrksn`) |
| `002050 LEAGUE CITY PKWY` vs `2050`; `03852` vs `3852` | Strip leading zeros from numbers | Same number, different formatting |
| `158ND AVE` vs `158th Avenue`; `1505 NINTH ST` vs `1505 9th Street` | Strip ordinal suffixes; map `one..ten`/`first..tenth` to digits; split digit/letter runs | Numbers are the most specific address features |
| `Vijay F0ods`, `Y0ga`, `Harb0r` (zero for o) | Replace `0` between letters with `o` | Previously split into junk tokens `f`, `ods` |
| `capitalreliablenetworks.com`, `#Bergersociety`, `vadodararealtors.com` | `c_` features: concatenation of the first 2 and first 3 name tokens, and any single token ≥ 8 chars | Run-together domain/hashtag names now equal the concatenated S1 name |
| `MORGANTNO` vs `Morganton`, `WRREN ST` vs `Warren St` | Phonetic skeleton on address words too (`ak_`) | Typos that keep consonant order collide |

**Speed change (needed for test scale):** v1 took ~100 s for 10K queries, which extrapolates to **~4.7 h** for the 1.7M test S1. Most of that went into re-weighting and transposing every chunk for every query batch. v2 adds pass 1b: rows are regrouped per country into shards of ≤2M rows that are **already IDF-weighted, normalised and transposed** on disk. A query batch now only loads a shard and multiplies. The raw hashed chunks are deleted after sharding. The IDF cut-off (`min_idf`) moved to index-build time as a result.

| Setup | Recall | Notes |
|---|---|---|
| v2 smoke test (2K queries vs sample pool, K=20) | 97.84% | v1: 97.51% |
| v2 full 10.3M | *running* | |

#### Engineering notes
- `pd.read_csv` on the full 5M-row files crashed (segfault with `dtype=str`, `MemoryError` via pyarrow) while RAM was low. Chunked reading (`chunksize`, `keep_default_na=False` so names like "NA" stay strings) fixes it.
- Files use standard CSV double-quote escaping (e.g. `"""ehpad Club SAS"`), so the default quote handling is correct.
- New deps: `anyascii`, `sparse_dot_topn`. Both are permissive licences and neither is an external data lookup.

---

## 5. Phase 4: Feature Engineering
**Status:** Up Next
**Logic & Decisions:**
- We now have ~844,000 candidate pairs. We need to calculate how similar Record A is to Record B.
- We will compute string distances (Levenshtein, Jaccard) for both the name and the address.
