# Amazon ML Challenge 2026: Implementation Plan

Given the scale of the dataset (2.4 GB) and the complexity of the Entity Resolution task, we need a highly systematic approach. We cannot load all the data into memory for naive comparisons. We will follow a robust 6-phase pipeline.

---

## Phase 1: Project Setup & Data Subsampling 
*The dataset is too large for fast iteration. We must create a representative, smaller dataset for rapid prototyping.*

1. **Initialize Project Structure:**
   - Create directories: `code/business_entity_resolution/src/`, `notebooks/`, and `data/sample/`.
   - Set up the environment (`requirements.txt`) with libraries like `pandas`, `scikit-learn`, `numpy`, and string matching libraries like `jellyfish` or `rapidfuzz`.
2. **Create a Representative Sample:**
   - Write a script to randomly sample ~10,000 entities from `train_source1.tsv`.
   - Use the `train_ground_truth.tsv` to find all corresponding true matches for these 10,000 entities in `Source 2` and `Source 3`.
   - Also, include a healthy amount of negative samples (records that don't match).
   - Save these as `sample_source1.tsv`, `sample_source2.tsv`, etc. We will use these samples for Phases 2, 3, and 4.

## Phase 2: Data Preprocessing & Normalization
*The data is noisy. We need to standardize it to improve our blocking and matching algorithms.*

1. **Text Normalization:**
   - Convert all text to lowercase.
   - Remove special characters and standardise spacing.
2. **Standardization:**
   - Handle abbreviations (e.g., expand "Pvt", "Ltd", "Corp" to "Private", "Limited", "Corporation").
   - Standardize addresses (e.g., "St", "Rd", "Ave" -> "Street", "Road", "Avenue").
3. **Handling Missing Values:**
   - Fill NaN values with empty strings or placeholder tokens.

## Phase 3: Candidate Generation (Blocking)
*Comparing every Source 1 record against every Source 2/3 record is computationally impossible ($O(N \times M)$). Blocking reduces this search space (Optimizes for Recall).*

1. **Design Blocking Keys:**
   - Create cheap, fast keys. Examples:
     - `Key 1`: First 3 letters of business name + ZIP code.
     - `Key 2`: Exact match on standardized business name.
     - `Key 3`: First word of business name + City.
2. **Generate Candidate Pairs:**
   - Group records from Source 1 and Sources 2/3 that share the exact same blocking keys.
   - If a pair shares a key, they become a "Candidate Pair".
3. **Evaluate Blocking Strategy:**
   - Calculate the **Recall** of our blocking stage on the sampled ground truth. (Did we accidentally filter out true matches?).
   - Calculate the **Reduction Ratio** (How much did we shrink the search space?).
   - Refine keys until we hit ~95%+ recall.
   - Export this to `candidate_pairs.tsv`.

## Phase 4: Feature Engineering
*Now we look closely at the candidate pairs and calculate how similar they actually are.*

For every candidate pair (Record A from Source 1, Record B from Source 2/3), compute similarity scores:
1. **Name Similarity Features:**
   - Levenshtein distance (edit distance).
   - Jaccard similarity (word overlap).
   - TF-IDF Cosine similarity.
2. **Address Similarity Features:**
   - Exact match boolean (True/False).
   - Token intersection (how many words match exactly in the address).
   - Numeric match (do the street numbers match?).
3. **Target Variable Setup:**
   - Label each candidate pair as `1` (True Match) or `0` (False Match) based on `train_ground_truth.tsv`.

## Phase 5: ML Model Training & Validation (Matching)
*Train a model to look at the features and decide the final match (Optimizes for Precision).*

1. **Model Selection:**
   - Train an XGBoost, LightGBM, or Random Forest classifier. Trees handle non-linear similarity features very well.
2. **Class Imbalance Handling:**
   - There will be far more false matches (0s) than true matches (1s) in our candidate set. Use techniques like scale_pos_weight or downsampling.
3. **Evaluation & Threshold Tuning:**
   - We evaluate on the **$F_{0.5}$ score**!
   - Because false merges are penalized twice as heavily, we will increase the probability threshold (e.g., only predict a match if probability > 0.7) to maximize precision.
   - Pay special attention to "Singletons" (Source 1 entities with no matches).

## Phase 6: Full Scale Execution & Inference
*Run the pipeline on the full datasets.*

1. **Apply to Full Train Data:** Run the optimized pipeline on the full 2.4GB data to train the final robust model. (This is where we might need AWS if local compute fails).
2. **Apply to Test Data:** 
   - Preprocess test data (remember to handle the new country: France).
   - Run Blocking to get test candidates.
   - Run the trained ML model on test candidates to get predictions.
3. **Format Output:**
   - Ensure exactly one row per Source 1 entity.
   - Aggregate matched IDs into comma-separated lists.
   - Ensure singletons have empty lists.
4. **Validation:**
   - Run `utils/validate_submission.py` to guarantee zero formatting errors.

## Phase 7: Final Packaging
1. Zip the `output/` folder (`matching_results.tsv`, `candidate_pairs.tsv`).
2. Zip the `code/` folder with `requirements.txt`.
3. Complete the `Documentation_template.md` describing our keys, features, and model.
