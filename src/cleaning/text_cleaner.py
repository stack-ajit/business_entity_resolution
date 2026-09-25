import pandas as pd
import re

def clean_text(text):
    """
    Lowercases text, removes special characters, and handles missing values.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def standardize_abbreviations(text):
    """
    Standardizes common business and address abbreviations.
    """
    if not text:
        return ""
        
    abbrev_dict = {
        # Business suffixes
        r'\bco\b': 'company',
        r'\binc\b': 'incorporated',
        r'\bcorp\b': 'corporation',
        r'\bltd\b': 'limited',
        r'\bpvt\b': 'private',
        r'\bllc\b': 'limited liability company',
        # Address parts
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bln\b': 'lane',
        r'\bct\b': 'court',
        r'\bsq\b': 'square',
        r'\bpl\b': 'place',
        r'\bhwy\b': 'highway',
        r'\bste\b': 'suite',
        r'\bapt\b': 'apartment',
        r'\bfl\b': 'floor'
    }
    
    for pattern, replacement in abbrev_dict.items():
        text = re.sub(pattern, replacement, text)
        
    return text

def preprocess_dataframe(df):
    """
    Applies text cleaning and standardization to business_name and business_address.
    """
    print("Preprocessing dataframe...")
    df = df.copy()
    
    for col in ['business_name', 'business_address']:
        if col in df.columns:
            # 1. Clean basic text
            df[f'{col}_clean'] = df[col].apply(clean_text)
            
            # 2. Standardize abbreviations
            df[f'{col}_clean'] = df[f'{col}_clean'].apply(standardize_abbreviations)
            
    return df

if __name__ == "__main__":
    # Quick test on the sample data
    sample_path = r"c:\Users\BIT\Downloads\amazon_ml_2026\data\sample\sample_source1.tsv"
    print(f"Loading {sample_path}")
    df = pd.read_csv(sample_path, sep='\t')
    
    # Preprocess
    df_clean = preprocess_dataframe(df)
    
    print("\nBefore vs After:")
    print(df_clean[['business_name', 'business_name_clean']].head(5))
    print("\n")
    print(df_clean[['business_address', 'business_address_clean']].head(5))
