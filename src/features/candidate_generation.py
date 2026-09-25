import pandas as pd
import numpy as np
import os
import sys

# Add src to path so we can import cleaning module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cleaning.text_cleaner import preprocess_dataframe

def generate_blocking_keys(df):
    """
    Creates multiple blocking keys for high recall candidate generation.
    """
    df = df.copy()
    
    # Extract common parts
    name_prefix = df['business_name_clean'].str.slice(0, 4)
    name_first_word = df['business_name_clean'].str.split().str[0]
    name_last_word = df['business_name_clean'].str.split().str[-1]
    
    addr_prefix = df['business_address_clean'].str.slice(0, 4)
    addr_first_word = df['business_address_clean'].str.split().str[0]
    addr_last_word = df['business_address_clean'].str.split().str[-1]
    
    # Generate 5 overlapping keys to maximize recall
    df['block_key_1'] = name_prefix + "_" + addr_first_word
    df['block_key_2'] = name_first_word + "_" + addr_last_word
    df['block_key_3'] = name_prefix + "_" + addr_prefix
    df['block_key_4'] = name_last_word + "_" + addr_first_word
    df['block_key_5'] = name_first_word + "_" + addr_first_word
    
    return df

def get_candidates(source1_df, other_df):
    """
    Merges Source 1 and Source 2/3 on blocking keys to generate candidate pairs.
    """
    candidates_list = []
    
    for i in range(1, 6):
        key = f'block_key_{i}'
        c = pd.merge(
            source1_df[['entity_id', key]], 
            other_df[['entity_id', key]], 
            on=key,
            suffixes=('_s1', '_s23')
        )
        candidates_list.append(c[['entity_id_s1', 'entity_id_s23']])
        
    all_candidates = pd.concat(candidates_list)
    all_candidates = all_candidates.drop_duplicates(subset=['entity_id_s1', 'entity_id_s23'])
    
    return all_candidates


if __name__ == "__main__":
    sample_dir = r"c:\Users\BIT\Downloads\amazon_ml_2026\data\sample"
    
    print("Loading samples...")
    s1 = pd.read_csv(os.path.join(sample_dir, "sample_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(sample_dir, "sample_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(sample_dir, "sample_source3.tsv"), sep="\t")
    
    # Preprocess
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    
    # Generate Keys
    print("Generating blocking keys...")
    s1 = generate_blocking_keys(s1)
    s2 = generate_blocking_keys(s2)
    s3 = generate_blocking_keys(s3)
    
    # Combine s2 and s3 for candidate generation
    s23 = pd.concat([s2, s3])
    
    # Get Candidates
    print("Generating candidate pairs...")
    candidates = get_candidates(s1, s23)
    print(f"Generated {len(candidates)} total candidate pairs.")
    
    # Evaluate Recall against Ground Truth
    print("Evaluating Recall...")
    gt = pd.read_csv(os.path.join(sample_dir, "sample_ground_truth.tsv"), sep="\t")
    
    true_pairs = set()
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids'])
        if matches and matches != 'nan':
            for s23_id in matches.split(','):
                true_pairs.add((s1_id, s23_id.strip()))
                
    generated_pairs = set(zip(candidates['entity_id_s1'], candidates['entity_id_s23']))
    
    found_true_pairs = true_pairs.intersection(generated_pairs)
    
    recall = len(found_true_pairs) / len(true_pairs) if len(true_pairs) > 0 else 0
    print(f"Total true matches in sample: {len(true_pairs)}")
    print(f"True matches found by blocking: {len(found_true_pairs)}")
    print(f"Blocking Recall: {recall:.2%}")
    
    # Reduction Ratio
    possible_pairs = len(s1) * len(s23)
    reduction = 1 - (len(candidates) / possible_pairs)
    print(f"Reduction Ratio: {reduction:.4%}")
