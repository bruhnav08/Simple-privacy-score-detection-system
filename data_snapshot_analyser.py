# data_snapshot_analyser.py

import pandas as pd

print("--- Starting Data Snapshot Analysis ---")

# --- 1. Analyze socialmedia.csv (if available) ---
# Note: The original 'socialmedia.csv' was mentioned as a basis for synthetic data,
# but it was not provided in the initial files. We'll try to load it,
# but it's okay if it's not found, as extended_socialmedia.csv is our primary base.
print("\n--- Analyzing socialmedia.csv (Original Base Data) ---")
try:
    # Attempt to load the original socialmedia.csv - it might not be in your provided files
    df_original = pd.read_csv('socialmedia.csv')
    print("socialmedia.csv loaded successfully.")
    print(f"Shape: {df_original.shape}")
    print("\nHead (first 5 rows):")
    print(df_original.head())
    print("\nInfo (column types and non-null counts):")
    df_original.info()
except FileNotFoundError:
    print("socialmedia.csv not found. This is expected if it was just a conceptual basis for synthetic data.")
    print("Proceeding without original socialmedia.csv.")
except Exception as e:
    print(f"An error occurred while loading socialmedia.csv: {e}")

# --- 2. Analyze extended_socialmedia.csv (Synthetic Data) ---
print("\n--- Analyzing extended_socialmedia.csv (Synthetic Data) ---")
try:
    df_extended = pd.read_csv('extended_socialmedia.csv')
    print("extended_socialmedia.csv loaded successfully.")
    print(f"Shape: {df_extended.shape}")
    print("\nHead (first 5 rows):")
    print(df_extended.head())
    print("\nInfo (column types and non-null counts):")
    df_extended.info()
    print("\nUnique Platforms in extended_socialmedia.csv:")
    print(df_extended['Platform'].unique())
except FileNotFoundError:
    print("Error: 'extended_socialmedia.csv' not found.")
    print("Please ensure 'generate_synthetic_data.py' has been run to create this file.")
    exit()
except Exception as e:
    print(f"An error occurred while loading extended_socialmedia.csv: {e}")
    exit()

# --- 3. Analyze preprocessed_socialmedia.csv (Enhanced Preprocessed Data) ---
print("\n--- Analyzing preprocessed_socialmedia.csv (Enhanced Preprocessed Data) ---")
try:
    df_preprocessed = pd.read_csv('preprocessed_socialmedia.csv')
    print("preprocessed_socialmedia.csv loaded successfully.")
    print(f"Shape: {df_preprocessed.shape}")
    print("\nHead (first 5 rows):")
    print(df_preprocessed.head())
    print("\nInfo (column types and non-null counts):")
    df_preprocessed.info()
    print("\nSample of new features (first 5 rows):")
    # Try to print some of the newly added columns to confirm their presence and values
    new_features_sample = [
        'has_email_in_bio', 'bio_sentiment_score', 'post_text_sentiment_score',
        'days_since_post_normalized', 'shares_to_likes_ratio_normalized'
    ]
    # Check if these columns exist before trying to print them
    existing_new_features = [f for f in new_features_sample if f in df_preprocessed.columns]
    if existing_new_features:
        print(df_preprocessed[existing_new_features].head())
    else:
        print("No specific new features found in sample, but all columns are listed above.")

except FileNotFoundError:
    print("Error: 'preprocessed_socialmedia.csv' not found.")
    print("Please ensure '1.py' has been run successfully to create this file.")
    exit()
except Exception as e:
    print(f"An error occurred while loading preprocessed_socialmedia.csv: {e}")
    exit()

print("\n--- Data Snapshot Analysis Complete ---")