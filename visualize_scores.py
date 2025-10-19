# visualize_scores.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # For nicer plots
import numpy as np # For potential nan handling

print("--- Starting Enhanced Score Visualization ---")

# --- Load the preprocessed data (to get proxy_risk_score and Platform info) ---
try:
    df_full = pd.read_csv('preprocessed_socialmedia.csv')
    print("Preprocessed data loaded successfully.")
except FileNotFoundError:
    print("Error: 'preprocessed_socialmedia.csv' not found.")
    print("Please ensure '1.py' has been run successfully.")
    exit()
except Exception as e:
    print(f"An error occurred while loading preprocessed_socialmedia.csv: {e}")
    exit()

# --- Load the comprehensive privacy scores (predictions) ---
try:
    df_predicted_scores = pd.read_csv('users_with_comprehensive_privacy_scores.csv')
    print("Predicted scores loaded successfully from 'users_with_comprehensive_privacy_scores.csv'.")
except FileNotFoundError:
    print("Error: 'users_with_comprehensive_privacy_scores.csv' not found.")
    print("Please ensure 'direct_privacy_scorer.py' has been run successfully.")
    exit()
except Exception as e:
    print(f"An error occurred while loading predicted scores: {e}")
    exit()

# --- Load Feature Importances ---
try:
    df_importances = pd.read_csv('direct_ml_feature_importances.csv')
    print("Feature importances loaded successfully from 'direct_ml_feature_importances.csv'.")
except FileNotFoundError:
    print("Error: 'direct_ml_feature_importances.csv' not found.")
    print("Please ensure 'direct_privacy_scorer.py' has been run successfully.")
    exit()
except Exception as e:
    print(f"An error occurred while loading feature importances: {e}")
    exit()


# --- Re-calculate proxy_risk_score in df_full for plotting actual vs predicted ---
# This logic comes directly from your direct_privacy_scorer.py
df_full['proxy_risk_score'] = (
    df_full['has_contact_info'] +
    df_full['has_sensitive_keywords'] +
    df_full['is_specific_location'] +
    df_full['privacy_setting_score'] +
    df_full['followers_normalized']
)
min_score = df_full['proxy_risk_score'].min()
max_score = df_full['proxy_risk_score'].max()
if (max_score - min_score) > 0:
    df_full['proxy_risk_score'] = (df_full['proxy_risk_score'] - min_score) / (max_score - min_score)
else:
    df_full['proxy_risk_score'] = 0.0

# Merge predicted scores back to df_full based on 'User ID' for integrated plotting
df_full = pd.merge(df_full, df_predicted_scores[['User ID', 'comprehensive_privacy_score']], on='User ID', how='left')


# --- Begin Plotting ---
plt.style.use('seaborn-v0_8-darkgrid') # Apply a nice style

fig, axes = plt.subplots(3, 2, figsize=(18, 18)) # Create a 3x2 grid of plots

# Plot 1: Histogram of Comprehensive Privacy Scores
sns.histplot(df_full['comprehensive_privacy_score'], bins=30, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('1. Distribution of Comprehensive Privacy Scores')
axes[0, 0].set_xlabel('Comprehensive Privacy Score (0-1)')
axes[0, 0].set_ylabel('Number of Users')
axes[0, 0].axvline(x=0.3, color='r', linestyle='--', label='Low Risk < 0.3') #
axes[0, 0].axvline(x=0.6, color='orange', linestyle='--', label='Medium Risk > 0.6') #
axes[0, 0].legend()


# Plot 2: Violin Plot of Scores by Platform
sns.violinplot(x='Platform', y='comprehensive_privacy_score', data=df_full, ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('2. Comprehensive Privacy Score Distribution by Platform')
axes[0, 1].set_xlabel('Platform')
axes[0, 1].set_ylabel('Comprehensive Privacy Score (0-1)')
axes[0, 1].tick_params(axis='x', rotation=45) # Rotate x-axis labels for readability


# Plot 3: Scatter Plot of Predicted vs. Actual (Proxy) Scores
sns.scatterplot(x='proxy_risk_score', y='comprehensive_privacy_score', data=df_full, ax=axes[1, 0], alpha=0.7, color='purple')
axes[1, 0].set_title('3. Predicted vs. Actual (Proxy) Privacy Scores')
axes[1, 0].set_xlabel('Actual (Proxy) Privacy Score')
axes[1, 0].set_ylabel('Predicted Comprehensive Privacy Score')
axes[1, 0].plot([0, 1], [0, 1], color='red', linestyle='--', label='Ideal Prediction Line') # Add ideal line
axes[1, 0].legend()


# Plot 4: Top 10 Feature Importances Bar Chart
sns.barplot(x='Importance', y='Feature', data=df_importances.head(10), ax=axes[1, 1], palette='plasma')
axes[1, 1].set_title('4. Top 10 Feature Importances for Comprehensive Privacy Score')
axes[1, 1].set_xlabel('Importance (Contribution to Score)')
axes[1, 1].set_ylabel('Feature')


# Plot 5 & 6: Additional specific insights (e.g., specific feature vs score)
# Example: Relationship between privacy_setting_score and comprehensive_privacy_score
sns.boxplot(x='privacy_setting_score', y='comprehensive_privacy_score', data=df_full, ax=axes[2, 0], palette='coolwarm')
axes[2, 0].set_title('5. Score by Privacy Setting (0=Private, 1=Public)')
axes[2, 0].set_xlabel('Privacy Setting Score')
axes[2, 0].set_ylabel('Comprehensive Privacy Score')

# Example: Relationship between has_contact_info and comprehensive_privacy_score
sns.boxplot(x='has_contact_info', y='comprehensive_privacy_score', data=df_full, ax=axes[2, 1], palette='autumn')
axes[2, 1].set_title('6. Score by Presence of Contact Info (0=No, 1=Yes)')
axes[2, 1].set_xlabel('Has Contact Info')
axes[2, 1].set_ylabel('Comprehensive Privacy Score')


plt.tight_layout() # Adjust layout to prevent overlapping
plt.show() # Display all plots

print("\n--- Enhanced Score Visualization Complete ---")
print("A series of plots visualizing your privacy scores and feature importances should have appeared.")