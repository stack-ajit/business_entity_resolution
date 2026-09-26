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
| v2 full 10.3M, K=20 | 88.41% | **worse** than v1 (90.55%) |
| v2 full 10.3M, K=50 | 92.51% | worse than v1 (93.20%) |
| v2 full 10.3M, K=100 | 94.46% | ≈ v1 (94.35%) |

v2 by country at K=100: India 91.4% (v1 92.2%, **down**), US 96.5% (v1 95.8%, up). **Query time: 7 s for 10K queries (v1: ~100 s)**, so the 1.7M test S1 records take ~20 min instead of ~4.7 h. The speed redesign clearly works.

**Takeaway: v2 as a bundle hurts the top of the ranking.** Hypotheses:
- The phonetic key is much coarser than the vowel-drop skeleton (m=n, g=j, v=w=b). Short Indian name tokens collide (`vijay`→`bj`, `guru`→`jr`), which lets unrelated records into the top-K. That would explain why India dropped.
- `ak_` duplicates every address word, doubling the address block's weight relative to the name.
- Extra `c_` features dilute cosine for normal names.

Rather than guess, the next step is an ablation, not another bundle. **Lesson: change one feature group at a time and measure.**

Also fixed: v2 hashing was 3.4× slower (18 regex passes per token). Memoising `skeleton` (`lru_cache`; tokens repeat heavily) and skipping `anyascii` for pure-ASCII strings roughly halves hashing time.

### Phase 3 — Ablation harness
`scratchpad/ablate.py` (to be moved into `src/blocking/` if kept): the 10K sample S1 against their true matches plus ~1.5M random train S2/S3 distractors, all in memory. Each variant takes minutes instead of ~25 min for a full rebuild. Absolute recall is higher than at full scale (fewer distractors), but **relative** ordering of variants is what matters. Also tests **multi-view retrieval**: separate top-K lists from the combined, name-only and address-only vectors, unioned under the same total budget K. The idea is that a record with a missing address or a renamed business still gets its own shortlist slots.
Moved into the repo as `src/blocking/ablate_blocking.py`.

#### Ablation results (recall / candidate pairs, 10K queries, ~1.5M-record pool)
| Variant | @10 | @20 | @30 | @50 |
|---|---|---|---|---|
| v1 (baseline) | 93.08% | 94.61% | 95.25% | 96.00% |
| + number normalisation (v2 norm) | 93.32% | 94.84% | 95.50% | 96.16% |
| + v2 norm + phonetic skeleton v2 | 93.44% | 94.95% | 95.57% | 96.22% |
| + v2 norm + skeleton v1 **and** v2 | 92.13% | 93.98% | 94.71% | 95.57% |
| + v2 norm + `c_` concat features | **90.91%** | 94.20% | 95.28% | 96.23% |
| + v2 norm + `ak_` address skeleton | 93.74% | 95.15% | 95.70% | 96.32% |
| v2 (all of the above) | 92.23% | 94.83% | 95.67% | 96.44% |
| **final: v2 norm + skeleton v2 + `ak_`** | **93.82%** | **95.20%** | **95.72%** | 96.33% |
| v1 + views (combined ½K, name ¼K, addr ¼K) | 87.90% (59K) | 94.23% (131K) | 95.10% (197K) | 95.90% (339K) |
| v2 norm + views | 88.44% (59K) | 94.64% (131K) | 95.39% (197K) | 96.16% (340K) |

**Conclusions:**
- My hypothesis about the phonetic key was **wrong**. The coarse phonetic skeleton *helps* (+0.1). The v2 regression came from the **`c_` concat features (−2.4 pts @10)**: every ordinary 2–3 word name gets extra rare-looking features that match unrelated records. They are **removed**, even though they fixed a few domain-name cases.
- Using both skeletons double-counts the name and hurts (−1.2), so only v2 is kept.
- `ak_` helps (+0.4 @10), because address typos (`morgantno`, `wrren`) are common.
- **Views** don't raise recall per unit of K. But because the union deduplicates, they reach similar recall with **~30% fewer pairs** (96.16% with 340K pairs against 497K). That is an efficiency option for later, not adopted now, to keep the pipeline simple.
- **Final blocking = v2 normalisation + phonetic skeleton v2 + `ak_`, no `c_`.** It is best or tied-best at every K, ~+0.7 pts over v1 at K=10.
- The local `data/cache/train_index` was built with `c_`, so it is stale. The Kaggle run rebuilds from scratch with the final features.

#### Engineering notes
- `pd.read_csv` on the full 5M-row files crashed (segfault with `dtype=str`, `MemoryError` via pyarrow) while RAM was low. Chunked reading (`chunksize`, `keep_default_na=False` so names like "NA" stay strings) fixes it.
- Files use standard CSV double-quote escaping (e.g. `"""ehpad Club SAS"`), so the default quote handling is correct.
- New deps: `anyascii`, `sparse_dot_topn`. Both are permissive licences and neither is an external data lookup.

---

## 5. Decision: Move Forward + Run Heavy Stages on Kaggle
**Status:** Setup done, awaiting data upload + code push

**Why move forward now:** blocking recall went from 53% to ~93% at K=50 at full scale, and queries run in minutes. The remaining ablation tunes the last 1–2%, and the winning variant is a flag plus an index rebuild, with no downstream code changes. With a 27 Sep deadline and no submission yet, an end-to-end pipeline matters more than blocking polish.

**Why Kaggle:** the laptop has 16 GB RAM shared with the browser (at one point only 0.34 GB was free and pandas crashed). A Kaggle CPU session has ~30 GB RAM.
- **Trade-off:** Kaggle gives 4 cores against the laptop's 12, so CPU-bound steps (feature hashing, rapidfuzz) run slower there.
- **No GPU needed:** nothing in this pipeline benefits from a GPU. LightGBM and rapidfuzz are CPU-bound.

**Changes to make the code portable:**
- `src/config.py`: all paths come from `ER_DATA_DIR` (folder with `train/`, `test/`) and `ER_WORK_DIR` (writable caches/outputs). Local defaults are unchanged. The hard-coded `c:\Users\...` paths in `create_sample.py` are removed.
- `requirements.txt`: pinned versions (also required for the final submission zip).
- `kaggle/er_pipeline.ipynb`: installs deps, clones the GitHub repo, auto-locates the uploaded dataset, sets the env vars, then runs the stages. Each stage **skips itself if its output exists**, so a dead session can resume.

---

### Kaggle run #1: final blocking confirmed at full scale (10K queries vs 10.3M train S2/S3)
| K | v1 (local) | **final (Kaggle)** |
|---|---|---|
| 5 | 77.90% | **79.59%** |
| 10 | 86.91% | **88.39%** |
| 20 | 90.55% | **91.64%** |
| 30 | 92.03% | **92.88%** |
| 50 | 93.20% | **93.92%** |
| 75 | 93.97% | **94.54%** |
| 100 | 94.35% | **94.94%** |

At K=100: India 91.64%, US 97.07%; S2 95.16%, S3 94.73%. The ablation ranking held at full scale: +0.6 to +1.5 pts at every K. Kaggle timings: index build 7.5 min (43 chunks → 6 shards); 10K queries in 5 s.

**Chosen K for test = 50** (93.9% recall ceiling). K=100 adds only +1.0 pt for 2× the pairs and feature cost.

---

## 6. Phase 4: Matching Model: Pair Features + Training Set
**Status:** Training set built on Kaggle. 150K train S1 → **7,499,222 pairs, 487,198 positives**, blocking recall 93.91% (consistent with the evaluation). 736 s total (features: ~310 s for 7.5M pairs).
**Code:** `src/matching/pair_features.py`, `src/matching/build_training_set.py`

**Logic & Decisions:**
- **Train on real blocking output, not random negatives.** Take 150K random train S1 records, run the same top-K=50 blocking used at test time against the full train index, and label candidates from the ground truth. The model learns to reject the look-alikes it will actually face (same street, similar name), which random negatives would not teach.
- **No `country` feature.** Test has France, which never appears in training. All features are country-agnostic similarities, so the model transfers.
- **Features (35):**
  - *retrieval context*: score, rank, score ÷ best score for that S1, gap to best, number of candidates. "Is this the best candidate for this entity?" matters for precision.
  - *name*: rapidfuzz ratio, token-set, token-sort, partial, Jaro-Winkler on the full name; the same on the **core name** (legal suffixes like pvt/ltd/llc/sas/sarl removed); on the concatenated core (domains/hashtags); and on the phonetic skeleton (transliterated names); token Jaccard, token counts.
  - *address*: ratio, token-set, partial, token Jaccard.
  - *numbers*: Jaccard of number sets, count shared, **conflict flag** (both have numbers but none shared, a strong negative), longest shared number, postcode (5/6-digit) match / conflict.
  - *flags*: empty address (either side), non-Latin candidate name, source S2 vs S3.
- Entities with zero candidates are kept in a separate file: they still count in macro F0.5 (an empty prediction scores 1.0 for singletons and 0 otherwise).
- **Next:** LightGBM (MIT) with an S1-grouped train/validation split. Then choose per-entity selection rules (probability threshold, max matches, relative-to-best cut-off) by directly maximising **macro F0.5 on validation**, including empty predictions.

---

## 7. Phase 5: Model Training, Selection Rule, Test Inference
**Status:** Code written and smoke-tested end-to-end locally (validator **PASS**); first real run pending on Kaggle
**Code:** `src/matching/train_matcher.py`, `src/matching/selection.py`, `src/matching/predict_test.py`

**Logic & Decisions:**
- **Split by S1 entity (80/20), not by row.** All candidates of an entity stay on one side, like unseen test entities. A row-level split would leak entity-level information and inflate validation.
- **LightGBM** (MIT, CPU): 127 leaves, lr 0.05, early stopping on validation log-loss. It is refit on 100% of entities with 1.1× the best iteration count.
- **Selection rule is tuned on the official metric, not on AUC/log-loss.** The rule keeps a candidate if `p ≥ threshold` AND `p ≥ rel × best p of that entity`, capped at `max_matches`. It is grid-searched to maximise **macro F0.5 over all validation entities**, including entities with no candidates (counted as misses or correct singletons). The same `selection.py` code runs at test time, so validation and test apply exactly the same rule.
- **Diagnostics printed** so the score can be explained:
  - the *oracle* F0.5 (perfect classifier on our candidates): the ceiling set by blocking;
  - the *all-empty* baseline: what predicting only singletons earns (5.6%);
  - *top-1 with p ≥ 0.5*: shows that picking only one match loses a lot, because entities average 3.5 matches.
- **Streaming test inference.** 1.7M S1 × 50 = ~86M pairs cannot be held in memory. S1 is processed in 100K batches: retrieve → features → predict → select. S2/S3 raw strings are read once (`load_raw`) and prepared lazily per batch for only the candidate IDs.
- **Outputs follow the spec**: `candidate_pairs.tsv` = exactly the pairs the model scored (the last stage before the model, as the rules require); `matching_results.tsv` ⊆ candidates. Both have one row per S1 entity, empty when nothing. The official `validate_submission.py` runs automatically at the end.
- **Smoke test** (sample data standing in for both train and test; numbers only prove the plumbing): the validator passes, and 10,000 rows are written to both files.

### Kaggle run #2: first trained model (validation = 30K held-out train S1 entities)
| Metric | Value |
|---|---|
| **Validation macro F0.5 (tuned rule)** | **0.9425** |
| Oracle F0.5 (perfect classifier on our candidates) | 0.9770 |
| All-empty baseline (predict only singletons) | 0.0538 |
| Top-1 match with p ≥ 0.5 | 0.6798 |
| LightGBM best iteration (lr 0.05, early stopping) | 1864, val log-loss 0.01348 |
| Tuned rule | threshold 0.65, rel 0.7, max_matches 10 |

**Reading the numbers:**
- **Where the lost 5.75 pts go:** 2.3 pts are lost in blocking (true matches never retrieved caps us at 0.977); 3.45 pts are classifier/selection errors. Both are worth attacking, the classifier gap more so.
- **Selection rule barely matters.** All top-10 rules are within 0.0004. Thresholds of 0.65–0.70 are best: a slightly conservative cut, as expected for F0.5. `rel` has almost no effect and `max_matches` always saturates at 10. So the gains will come from better probabilities, not smarter cut-offs.
- **Top-1 only scores 0.68.** Entities average 3.5 matches, so returning several matches per entity is essential.
- **Feature importance:** retrieval `score` (31%) and `rank` (29%) dominate, followed by `core_partial` (7.9%), `num_jacc` (7.6%), `addr_tset` (5.5%) and `num_conflict` (2.1%). `postcode_*`, `addr_empty_*` and `n_cands` contribute ~0. `n_cands` is constant (always 50), and the postcode signal is already captured by `num_*`.

### Kaggle run #2: test inference exposed two operational problems
1. **Speed:** ~830 s per 100K-S1 batch (≈5M pairs) → **~4 h** for 1.7M test S1. String similarities ran on 1 core, and prediction uses ~2,050 trees.
2. **Draft sessions die.** The interactive session showed "Draft Session Starting..." mid-run, meaning it restarted and `/kaggle/working` was lost. Interactive sessions stop when the browser is idle.

**Fixes:**
- **Parallel features:** `pair_features` now splits pairs across all cores with a `fork` process pool. Records are inherited copy-on-write, so nothing is pickled going in. If the pool fails, it falls back to 1 process so a long run is never lost.
- **Per-batch checkpoints:** each finished 100K batch is saved to `cache/test_batches_k50_b100000/`. Re-running skips finished batches and only assembles the final TSVs. Verified locally: a rerun resumes instantly and the validator still passes.
- **Step timings** logged per batch (query / features / predict seconds) to find the next bottleneck.
- **Notebook reworked for "Save & Run All (Commit)"**, which runs headless for up to 12 h without the browser. The full-scale recall evaluation is now optional (`RUN_BLOCKING_EVAL`); `build_train_index.py` builds only the index. All steps use `python -u` so logs stream live. A final cell deletes the indexes to keep the saved output small.

### Next ideas (ranked by expected gain)
1. **One-to-one constraint / reverse competition.** S1 is deduplicated, so each S2/S3 record belongs to at most one S1 entity. At test time every S1 is scored, so an S2/S3 record claimed by several S1 entities can be given only to its highest-p S1. This should cut false merges between look-alike S1 entities (same street, similar names) and directly helps precision.
2. **Error analysis on validation:** split false positives and false negatives by country and source, and inspect the worst cases to design targeted features.
3. **K / model trade-off:** try a larger learning rate (fewer trees → faster inference) and drop the zero-importance features.

### Fix: the parallel feature code could exhaust memory (Kaggle run #3, second account)
**Symptom:** during `build_training_set` ("computing features" on 7.5M pairs), the notebook showed `IOStream.flush timed out` and stalled.
**Cause (my bug in the previous change):**
1. **Fork + shared dicts is not really shared.** With `fork`, every worker that *reads* a record touches its Python refcount, which copies that memory page. With 4 workers scoring random candidates, each ends up copying most of the multi-GB record dictionaries.
2. **Python tuples of floats are huge.** 7.5M rows × 30 floats as Python objects is ~7 GB, and the pool version held it twice (per-worker parts plus the flattened list).

**Fix:**
- Workers never touch the big dictionaries. The parent sends each worker only the small chunk of record tuples it scores (20K pairs).
- Work is fed in **bounded waves** (2 chunks per worker). `Pool.imap` would drain the generator into its queue immediately and rebuild the memory problem.
- Workers return **float32 arrays** (~120 bytes/row instead of ~1 KB). The serial path uses the same code.
- Portable (spawn or fork). Verified on Windows: **parallel output is identical to serial**. On 514K pairs: serial 31.1 s, 4 processes 17.9 s (1.7×), 8 processes 13.5 s. Memory stays flat.
- Notebook: `%cd /kaggle/working` before deleting and re-cloning the repo, which removes the harmless `shell-init: getcwd` warnings.

### Kaggle run #3: FIRST COMPLETE TEST PREDICTION (validator PASS)
Account `stackajit02`, code `d7ae31f` (memory-safe parallel features).

| Item | Value |
|---|---|
| Test S1 entities | 1,732,544 (both files, one row each) |
| `candidate_pairs.tsv` | 1,732,437 non-empty, **107 empty** (no candidate retrieved); 1.14 GB |
| `matching_results.tsv` | 1,622,991 non-empty, **109,553 empty = 6.32% predicted singletons**; 91 MB |
| Official validator | **PASS** (ID-existence check still to run with `--check-ids`) |
| Test index | France 1.43M records (1 shard), India 4.72M (3), US 3.82M (2): 9,969,589 S2/S3 |
| Time | ~2.8 h for 18 batches of 100K S1 |

**Per-batch timing** (100K S1 ≈ 5.0M pairs): query ~20 s, features ~205 s (was single-core before; 4 processes now), **predict ~330 s**. Prediction with ~2,050 LightGBM trees is now the bottleneck. Next speed lever: a higher learning rate (fewer trees) and/or dropping zero-importance features.

**Sanity:** predicted singleton rate 6.32% vs 5.6% true singletons in train. That is slightly conservative, which is desirable under precision-heavy F0.5, and plausible given ~6% of true matches are lost in blocking. The rate is stable across batches (~93.5K of 100K entities get matches in every batch), so there is no drift across the file.

**Open question:** France (1.43M S2/S3 records; never seen in training). The portal score vs validation 0.9425 will show whether the country-agnostic features transfer.

**Next:** download outputs (gzip first), submit `matching_results.tsv` to the portal, and compare the leaderboard score with validation. Then run validation error analysis and try the one-to-one constraint.
