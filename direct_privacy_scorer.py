# direct_privacy_scorer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("Starting the Comprehensive Privacy Score calculation using Machine Learning...\n")

# --- Step 2: Load Your Preprocessed Data ---
print("--- Step 2: Loading Preprocessed Data ---")
try:
    df = pd.read_csv('preprocessed_socialmedia.csv') # Load the preprocessed data
    print("\nData loaded successfully! Here's a peek at the first 5 rows:")
    print(df.head()) # Shows the first few rows of your data
    print("\nAnd some basic info about our data (data types, non-null counts):")
    df.info() # Gives a summary of columns, how many non-empty values, and data types
except FileNotFoundError:
    print("Error: 'preprocessed_socialmedia.csv' not found.")
    print("Please make sure the file is in the same directory as 'direct_privacy_scorer.py'.")
    exit() # Stop the script if the file isn't found
except Exception as e:
    print(f"An unexpected error occurred while loading the file: {e}")
    exit()

# --- Step 3: Define Your Target Variable (y) ---
print("\n--- Step 3: Defining the Target Variable (proxy_risk_score) ---")

# Calculate the proxy_risk_score, which will be our 'y' (what we predict)
# This definition comes directly from your derive_weights.py logic
df['proxy_risk_score'] = (
    df['has_contact_info'] +
    df['has_sensitive_keywords'] +
    df['is_specific_location'] +
    df['privacy_setting_score'] +
    df['followers_normalized']
)

# Normalize the proxy_risk_score to ensure it's between 0 and 1
# This is important if you want your final comprehensive score to be in a consistent range
min_score = df['proxy_risk_score'].min()
max_score = df['proxy_risk_score'].max()

if (max_score - min_score) > 0: # Check to prevent division by zero if all scores are identical
    df['proxy_risk_score'] = (df['proxy_risk_score'] - min_score) / (max_score - min_score)
else:
    df['proxy_risk_score'] = 0.0 # Assign 0 if all scores are the same (no variation)

y = df['proxy_risk_score'] # 'y' is now our prediction target

print("\n'proxy_risk_score' (our target variable 'y') calculated and normalized.")
print("Here's a peek at the first 5 values of 'y':")
print(y.head())
print("\n--- Summary Statistics for Proxy Risk Score (our 'temp privacy score') ---")
print(y.describe()) # <--- NEW LINE ADDED HERE
print("----------------------------------------------------------------------")


# --- Step 4: Define Your Features (X) ---
print("\n--- Step 4: Defining Features (X) and Handling Categorical Data ---")

# UPDATED LIST OF FEATURES_TO_INCLUDE based on your NEW preprocessed_socialmedia.csv columns
features_to_include = [
    'followers_normalized',
    'following_normalized', # Now present
    'likes_normalized',     # Now present
    'comments_normalized',  # Now present
    'engagement_normalized',
    'account_age_normalized', # Now present
    'is_verified',
    'activity_normalized',
    'hashtag_score',
    'media_type_score',
    'privacy_setting_score',
    'has_contact_info',
    'is_specific_location',
    'has_sensitive_keywords',
    'follower_following_ratio_normalized', # Now present
    'comments_per_post_normalized', # Now present
    # Note: bio_text_length_normalized and network_size_normalized were not in your final df.info() output, so omitted here.
    # If you later add them through 1.py, remember to add them here!

    # NEW Advanced Features confirmed in your output
    'has_email_in_bio',
    'has_phone_in_bio',
    'bio_sentiment_score',
    'post_text_sentiment_score',
    'sensitive_keyword_density_bio_normalized',
    'sensitive_keyword_density_post_normalized',
    'post_text_length_normalized',
    'days_since_post_normalized',
    'avg_time_between_posts_days_normalized',
    'shares_to_likes_ratio_normalized',
    'num_mentions_in_posts_normalized'
]


# Perform One-Hot Encoding for the 'Platform' column
df_encoded = pd.get_dummies(df, columns=['Platform'], prefix='Platform', drop_first=True)

# Combine numerical features with the new one-hot encoded platform features
# Check if all features_to_include actually exist after encoding/merging
# This step ensures we only try to select columns that are truly in the DataFrame
final_features_list = []
for feature in features_to_include:
    if feature in df_encoded.columns:
        final_features_list.append(feature)
    else:
        print(f"Warning: Feature '{feature}' not found in the DataFrame and will be skipped.")

# Add platform columns
final_features_list.extend([col for col in df_encoded.columns if col.startswith('Platform_')])

X = df_encoded[final_features_list]

print(f"\nFeatures (X) prepared. Total features: {len(X.columns)}.")
print("Here's a peek at the first 5 rows of our features (X):")
print(X.head())
print("\nList of all feature columns (X.columns):")
print(X.columns.tolist())

# --- Step 5: Split Your Data into Training and Testing Sets ---
print("\n--- Step 5: Splitting Data into Training and Testing Sets ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nData split: {len(X_train)} rows for training, {len(X_test)} rows for testing.")

# --- Step 6: Choose and Train Your Model ---
print("\n--- Step 6: Training the Machine Learning Model ---")
model = RandomForestRegressor(n_estimators=100, random_state=42)

print("Training the Random Forest model... This might take a moment depending on your data size.")
model.fit(X_train, y_train)
print("Model training complete!")

# --- Step 7: Evaluate Your Model ---
print("\n--- Step 7: Evaluating Model Performance on Test Data ---")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n--- Model Evaluation Results ---")
print(f"R-squared (R2 Score): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

if r2 > 0.6:
    print("\nModel shows good predictive power! R-squared is strong.")
elif r2 > 0.3:
    print("\nModel has learned something, but there's room for improvement. R-squared is moderate.")
else:
    print("\nModel performance is low. It might be predicting just the average, or worse.")

print("MAE tells us, on average, how far off our predictions are from the actual scores (on a 0-1 scale).")

# --- Step 8: Get Feature Importances (Your Justification!) ---
print("\n--- Step 8: Getting Feature Importances for Justification ---")
importances = model.feature_importances_
feature_names = X.columns
feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

print("Top 10 Most Important Features for Comprehensive Privacy Score:")
print(feature_importances_df.head(10))

output_importance_file = 'direct_ml_feature_importances.csv'
feature_importances_df.to_csv(output_importance_file, index=False)
print(f"\nFull feature importances saved to '{output_importance_file}'")

# --- Step 9: Use Your Model to Predict Scores for All Users ---
print("\n--- Step 9: Predicting Comprehensive Privacy Scores for All Users ---")
df['comprehensive_privacy_score'] = model.predict(X)

print("\nHere's a peek at the first 5 User IDs with their new Comprehensive Privacy Scores:")
print(df[['User ID', 'comprehensive_privacy_score']].head())

output_scores_file = 'users_with_comprehensive_privacy_scores.csv'
df[['User ID', 'comprehensive_privacy_score']].to_csv(output_scores_file, index=False)
print(f"\nAll user IDs with their Comprehensive Privacy Scores saved to '{output_scores_file}'")

print("\nProcess complete! Please review the output and the generated CSV files.")