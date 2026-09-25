import pandas as pd
import numpy as np
import os

def create_sample(data_dir, output_dir, sample_size=10000):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Ground Truth...")
    gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t")
    
    # Sample Source 1 IDs
    sampled_gt = gt.sample(n=sample_size, random_state=42)
    sampled_gt.to_csv(os.path.join(output_dir, "sample_ground_truth.tsv"), sep="\t", index=False)
    
    source1_ids = set(sampled_gt['source1_entity_id'])
    
    # Extract matching IDs from Source 2 and Source 3
    print("Extracting matching Source 2 & 3 IDs...")
    matched_ids = set()
    for ids_str in sampled_gt['matched_entity_ids'].dropna():
        if str(ids_str).strip():
            ids = [x.strip() for x in str(ids_str).split(',') if x.strip()]
            matched_ids.update(ids)
        
    print(f"Sampled {len(source1_ids)} Source 1 IDs.")
    print(f"Corresponding matches: {len(matched_ids)} IDs in Source 2/3.")
    
    # Process files
    for source in ["source1", "source2", "source3"]:
        print(f"Processing train_{source}.tsv...")
        # Read the file
        df = pd.read_csv(os.path.join(data_dir, f"train_{source}.tsv"), sep="\t", low_memory=False)
        
        if source == "source1":
            # Keep only the exact sampled Source 1 IDs
            sampled_df = df[df['entity_id'].isin(source1_ids)]
        else:
            # For source 2 and 3, keep the true matches
            true_matches_df = df[df['entity_id'].isin(matched_ids)]
            # Add noise (e.g. 50,000 random rows) to test our blocking algorithms
            noise_pool = df[~df['entity_id'].isin(matched_ids)]
            noise_sample_size = min(50000, len(noise_pool))
            noise_df = noise_pool.sample(n=noise_sample_size, random_state=42)
            sampled_df = pd.concat([true_matches_df, noise_df])
            
        sampled_df.to_csv(os.path.join(output_dir, f"sample_{source}.tsv"), sep="\t", index=False)
        print(f"Saved sample_{source}.tsv with {len(sampled_df)} rows.")

if __name__ == "__main__":
    TRAIN_DIR = r"c:\Users\BIT\Downloads\amazon_ml_2026\student_resource\dataset\train"
    OUT_DIR = r"c:\Users\BIT\Downloads\amazon_ml_2026\data\sample"
    create_sample(TRAIN_DIR, OUT_DIR, sample_size=10000)
    print("Sampling complete!")
