import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# --- Main Script ---

if __name__ == "__main__":
    input_file = 'preprocessed_socialmedia.csv' # Load the preprocessed file
    # This run will output general importances, not specific component weights yet
    output_importances_file = 'comprehensive_feature_importances.xlsx' 

    try:
        df = pd.read_csv(input_file)
        print(f"Loaded '{input_file}' with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Please ensure 1.py has been run on the extended data.")
        exit()

    # --- 1. Define the 'Proxy Privacy Risk Score' (your target variable) ---
    # Non-arbitrary proxy score, based on simple summation of inherently risky features.
    # Each component contributes equally based on its normalized/binary value.
    proxy_components = [
        'has_contact_info',
        'has_sensitive_keywords',
        'is_specific_location',
        'privacy_setting_score',
        'followers_normalized'
    ]

    df['raw_proxy_sum'] = df[proxy_components].fillna(0).sum(axis=1)
    df['proxy_risk_score'] = MinMaxScaler().fit_transform(df[['raw_proxy_sum']])


    print("Proxy privacy risk score calculated (non-arbitrary).")

    # --- 2. Prepare data for Machine Learning ---
    # This time, we include ALL engineered features from 1.py that could be relevant,
    # as we want to see their comprehensive importances.
    features_for_ml = [
        'followers_normalized',
        'engagement_normalized',
        'activity_normalized',
        'hashtag_score',
        'has_contact_info',
        'has_sensitive_keywords',
        'is_specific_location',
        'is_verified',
        'media_type_score',
        'privacy_setting_score',
        'bio_text_length_normalized', # NEW FEATURE
        'network_size_normalized'     # NEW FEATURE
    ]

    # Additional features from original data that were processed by 1.py but not normalized there:
    # 'account_age_days' is calculated but not normalized in 1.py
    # 'User ID', 'Post ID', 'User Following', 'User Engagement', 'User Interactions' are raw numbers
    
    scaler = MinMaxScaler()

    # Normalize these raw numerical columns if they exist and are not already normalized
    cols_to_normalize_for_ml_if_raw = [
        'account_age_days', # This one will be normalized as account_age_normalized
        'User ID',
        'Post ID',
        'User Following',
        'User Engagement',
        'User Interactions'
    ]

    for col in cols_to_normalize_for_ml_if_raw:
        if col in df.columns:
            # Create a normalized version of the column if it's not already in df (e.g., as 'col_normalized')
            # Handle potential non-numeric strings or NaNs before scaling
            if f"{col}_normalized" not in df.columns: # Prevent re-normalizing already normalized columns
                df[f"{col}_normalized"] = scaler.fit_transform(
                    pd.to_numeric(df[col], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
                )
            # Ensure the normalized version is in the features_for_ml list
            if f"{col}_normalized" not in features_for_ml:
                features_for_ml.append(f"{col}_normalized")
        # else: if raw column does not exist in df, it won't be added to features_for_ml.

    # Filter features_for_ml to only include those that exist in df
    # It's generally good practice to exclude raw IDs from ML input unless explicitly justified.
    # For now, we will INCLUDE them for *comprehensive* importances, but we might exclude them later.
    features_for_ml_final = [
        f for f in features_for_ml if f in df.columns
    ]

    # Use set to remove duplicates and sort for consistent order
    features_for_ml_final = sorted(list(set(features_for_ml_final)))


    # Ensure all features in final list exist and handle any NaNs within them
    X = df[features_for_ml_final]
    y = df['proxy_risk_score']

    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']: # Apply fillna only to numeric types
            X.loc[:, col] = X[col].fillna(X[col].mean())
        else:
            X.loc[:, col] = X[col].fillna(0) # For any potential object types that might still be there


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Data prepared for ML: X shape {X.shape}, y shape {y.shape}")
    print(f"Features used for ML: {features_for_ml_final}")

    # --- 3. Train a Machine Learning Model (Random Forest Regressor) ---
    print("Training RandomForestRegressor for comprehensive importances...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE on test set: {rmse:.4f}")

    # --- 4. Extract Comprehensive Feature Importances ---
    importances = model.feature_importances_
    comprehensive_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n--- Comprehensive Feature Importances (for all relevant engineered features) ---")
    print(comprehensive_importance_df)

    # --- 5. Save Comprehensive Importances ---
    with pd.ExcelWriter(output_importances_file) as writer:
        comprehensive_importance_df.to_excel(writer, sheet_name='Comprehensive_Importances', index=False)
    print(f"\nComprehensive feature importances saved to '{output_importances_file}'.")

    print("\nProcess Complete. Review 'comprehensive_feature_importances.xlsx' to decide feature groupings for DIS, RES, APS.")