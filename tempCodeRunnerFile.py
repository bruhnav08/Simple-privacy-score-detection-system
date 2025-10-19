You're absolutely right\! My apologies. The helper functions like `has_contact_info`, `is_specific_location`, etc., were moved into `derive_weights.py` to avoid duplication in `1.py`, but they are also needed in `generate_synthetic_data.py` to correctly calculate the probabilities (`prob_has_contact_info`, etc.) based on your original 39 rows.

The `NameError` occurs because `has_contact_info` function is called at line 140 within `generate_synthetic_data_based_on_existing`, but it's not defined within that script's scope.

To fix this, we need to include those necessary helper functions at the top of your `generate_synthetic_data.py` file.

Here's the corrected `generate_synthetic_data.py` with the helper functions added:

**Corrected File: `generate_synthetic_data.py`**

```python
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import re

# Initialize Faker
fake = Faker()

# --- Helper Functions (COPIED FROM 1.py / derive_weights.py FOR USE IN THIS SCRIPT) ---

EMAIL_RE = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
PHONE_RE = r'\b(?:\+?\d{1,3}[ -.]?)?(?:\(?\d{2,4}\)?[ -.]?)?\d{3,4}[ -.]?\d{3,4}\b'
URL_RE = r'(https?://|www\.)[A-Za-z0-9./?=_-]+'
def has_contact_info(text):
    if pd.isnull(text):
        return 0
    return int(bool(re.search(EMAIL_RE, str(text))) or bool(re.search(PHONE_RE, str(text))) or bool(re.search(URL_RE, str(text))))

sensitive_keywords = [
    'password', 'ssn', 'social security', 'credit card', 'debit card', 'address', 'phone', 'email', 'dob', 'date of birth',
    'passport', 'bank', 'account number', 'pin', 'security code', 'mother\'s maiden', 'confidential', 'private', 'secret',
    'salary', 'income', 'medical', 'disease', 'illness', 'diagnosis', 'treatment', 'insurance', 'license', 'driver', 'student id'
]
def has_sensitive_keywords(text):
    if pd.isnull(text):
        return 0
    text_lower = str(text).lower()
    return int(any(kw in text_lower for kw in sensitive_keywords))

GENERIC_LOCATIONS = {'unknown', 'global', 'worldwide', 'earth', 'planet', 'everywhere', 'anywhere', 'all', 'none', 'n/a', 'na', ''}
def is_specific_location(loc):
    if pd.isnull(loc):
        return 0
    return int(str(loc).strip().lower() not in GENERIC_LOCATIONS)

def extract_posts_per_week(activity):
    if pd.isnull(activity):
        return np.nan
    match = re.search(r'(\d+)\s*posts?\s*per\s*week', str(activity), re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r'(\d+)\s*post[s]?\s*per\s*day', str(activity), re.IGNORECASE)
    if match:
        return float(match.group(1)) * 7
    return np.nan
    
# Function to calculate account age days (needed for deriving stats if passed in original data)
CURRENT_DATE = datetime(2025, 7, 15)
def calc_account_age_days(date_str):
    try:
        return (CURRENT_DATE - datetime.strptime(str(date_str), '%d/%m/%Y')).days
    except Exception:
        return np.nan

# --- END Helper Functions ---


def generate_synthetic_social_media_data_based_on_existing(existing_df, num_rows=2000):
    """
    Generates a synthetic dataset for social media users and posts,
    with distributions based on an existing DataFrame.

    Args:
        existing_df (pandas.DataFrame): The DataFrame with existing data (your 40 rows).
        num_rows (int): The number of rows (users/posts) to generate.

    Returns:
        pandas.DataFrame: A DataFrame containing the synthetic data.
    """
    data = []

    # --- 1. Derive parameters from existing_df for improved accuracy ---

    # --- Categorical Features: Use actual distributions from existing data ---
    # Ensure no NaNs are passed to value_counts
    def get_categorical_distribution(series, default_options):
        cleaned_series = series.dropna().astype(str)
        if not cleaned_series.empty:
            counts = cleaned_series.value_counts(normalize=True)
            # Filter out any default_options that might not be in counts.index if default_options are generic.
            # This ensures weights sum to 1 and we only sample from observed categories.
            observed_options = list(counts.index)
            observed_weights = list(counts.values)
            return observed_options, observed_weights
        return default_options, [1/len(default_options)]*len(default_options) if default_options else ([],[])


    platforms, platform_weights = get_categorical_distribution(existing_df['Platform'], ['Instagram', 'Twitter', 'Facebook', 'LinkedIn', 'TikTok', 'Threads', 'Reddit', 'Other'])
    media_types, media_weights = get_categorical_distribution(existing_df['Media Type'], ['video', 'image', 'text', 'unknown'])
    privacy_settings_options, privacy_weights = get_categorical_distribution(existing_df['Privacy Settings'], ['public', 'private', 'friends only', 'limited', 'followers only'])
    account_verification_options, verification_weights = get_categorical_distribution(existing_df['Account Verification'], ['verified', 'not verified', 'yes', 'no', 'true', 'false'])


    # Date Range: Find min and max dates from existing data
    try:
        existing_dates = pd.to_datetime(existing_df['Account Creation Date'], format='%d/%m/%Y', errors='coerce').dropna()
        if not existing_dates.empty:
            min_date = existing_dates.min()
            max_date = existing_dates.max()
        else:
            min_date = datetime(2005, 1, 1)
            max_date = datetime(2024, 6, 30) # Default if no valid dates in existing
    except Exception:
        min_date = datetime(2005, 1, 1)
        max_date = datetime(2024, 6, 30)

    # Numerical Features: Estimate ranges/distributions
    # User Followers
    try:
        existing_followers = existing_df['User Followers'].astype(str).str.replace(',', '').astype(float).dropna()
        if not existing_followers.empty and existing_followers.min() > 0:
            log_followers = np.log(existing_followers[existing_followers > 0])
            followers_mean_log = log_followers.mean()
            followers_std_log = log_followers.std()
            if np.isnan(followers_std_log) or followers_std_log == 0:
                followers_std_log = 0.5 # Default small variance if all same
            if np.isnan(followers_mean_log):
                followers_mean_log = np.log(1000) # Default if all NaN
        else:
            followers_mean_log = np.log(1000)
            followers_std_log = 2.0
    except Exception:
        followers_mean_log = np.log(1000)
        followers_std_log = 2.0

    # Engagement: CRITICAL FIX - Derive typical engagement ratios from existing data
    try:
        existing_likes = existing_df['Likes/Reactions'].astype(str).str.replace(',', '').astype(float).dropna()
        existing_comments = existing_df['Comments'].astype(str).str.replace(',', '').astype(float).dropna()
        existing_shares = existing_df['Shares/Retweets'].astype(str).str.replace(',', '').astype(float).dropna()
        
        # Ensure that existing_followers, existing_likes are aligned for ratio calculation
        # Combine into a temporary df to ensure indices align
        temp_df = pd.DataFrame({
            'followers': existing_followers,
            'likes': existing_likes,
            'comments': existing_comments,
            'shares': existing_shares
        }).dropna()

        # Calculate average engagement as a percentage of followers, only for valid cases
        avg_likes_per_follower = (temp_df['likes'] / temp_df['followers']).mean() if (temp_df['followers'] > 0).any() else 0.05 # Default 5%
        avg_comments_per_like = (temp_df['comments'] / temp_df['likes']).mean() if (temp_df['likes'] > 0).any() else 0.1 # Default 10%
        avg_shares_per_like = (temp_df['shares'] / temp_df['likes']).mean() if (temp_df['likes'] > 0).any() else 0.05 # Default 5%
        
        # Scale the ratios to be more in line with observed values
        # The original mean was 1148 likes for ~10k followers -> ~11% engagement
        # So, if avg_likes_per_follower from your 40 rows is low (e.g., 0.0005), you need to adjust
        # For example, if original average likes per follower is 0.11, then set target range around that.
        
        # If your original 40 rows show 11% average likes (1148 likes / 10349 followers)
        # We need to ensure the min/max ratio bounds generate values around this observed mean.
        # Let's set a target average engagement ratio from the original data
        original_avg_engagement_rate = (existing_df['Likes/Reactions'].astype(str).str.replace(',', '').astype(float).mean() / existing_df['User Followers'].astype(str).str.replace(',', '').astype(float).mean()) if existing_df['User Followers'].astype(str).str.replace(',', '').astype(float).mean() > 0 else 0.11

        min_likes_ratio = max(0.001, original_avg_engagement_rate * 0.5) # Lower bound, but not zero
        max_likes_ratio = original_avg_engagement_rate * 1.5

        min_comments_ratio = max(0.01, avg_comments_per_like * 0.5)
        max_comments_ratio = avg_comments_per_like * 1.5
        min_shares_ratio = max(0.005, avg_shares_per_like * 0.5)
        max_shares_ratio = avg_shares_per_like * 1.5

        # Cap max ratios to prevent extremely unrealistic values (e.g., 100% engagement)
        max_likes_ratio = min(max_likes_ratio, 0.5) # Max 50% of followers for likes
        max_comments_ratio = min(max_comments_ratio, 0.4) # Max 40% of likes for comments
        max_shares_ratio = min(max_shares_ratio, 0.2) # Max 20% of likes for shares

    except Exception as e:
        print(f"Error deriving engagement ratios from existing data: {e}. Using robust defaults.")
        min_likes_ratio, max_likes_ratio = 0.01, 0.1
        min_comments_ratio, max_comments_ratio = 0.05, 0.2
        min_shares_ratio, max_shares_ratio = 0.02, 0.1


    # Posts per week: Match observed min/max
    try:
        existing_posts_per_week = existing_df['User Activity'].apply(extract_posts_per_week).dropna()
        if not existing_posts_per_week.empty:
            min_posts_per_week_observed = int(existing_posts_per_week.min())
            max_posts_per_week_observed = int(existing_posts_per_week.max())
            # Add some slight variance around observed min/max if they are too narrow
            min_posts_per_week = max(1, min_posts_per_week_observed - 1)
            max_posts_per_week = max_posts_per_week_observed + 1
        else:
            min_posts_per_week, max_posts_per_week = 1, 10 # Adjusted default if no data
    except Exception:
        min_posts_per_week, max_posts_per_week = 1, 10

    # Hashtags count range
    try:
        existing_hashtag_counts = existing_df['Hashtags'].astype(str).apply(lambda x: len(re.findall(r'#\w+', x))).dropna()
        if not existing_hashtag_counts.empty:
            min_hashtags = int(existing_hashtag_counts.min())
            max_hashtags = int(existing_hashtag_counts.max())
            if min_hashtags == max_hashtags: # Add some variance if all are same
                min_hashtags = max(0, min_hashtags - 1)
                max_hashtags = max_hashtags + 1
            if min_hashtags > max_hashtags: # Ensure min <= max after adjustments
                min_hashtags, max_hashtags = max_hashtags, min_hashtags
        else:
            min_hashtags, max_hashtags = 0, 5 # Reduced default range
    except Exception:
        min_hashtags, max_hashtags = 0, 5

    # Binary Feature Proportions (using helper functions directly)
    prob_has_contact_info = existing_df['User Bio'].apply(has_contact_info).mean() if not existing_df.empty else 0.0 # CRITICAL: If 0 in original, keep 0
    prob_has_sensitive_keywords = existing_df['Post Text'].apply(has_sensitive_keywords).mean() if not existing_df.empty else 0.05
    prob_is_specific_location = existing_df['Location'].apply(is_specific_location).mean() if not existing_df.empty else 0.95 # CRITICAL: If 1.0 in original, set high


    # Location: Derived from existing data
    generic_locations_set = {'unknown', 'global', 'worldwide', 'earth', 'planet', 'everywhere', 'anywhere', 'all', 'none', 'n/a', 'na', ''}
    existing_locations_list = existing_df['Location'].dropna().unique().tolist()
    specific_existing_locations = [loc for loc in existing_locations_list if str(loc).strip().lower() not in generic_locations_set]
    generic_existing_locations = [loc for loc in existing_locations_list if str(loc).strip().lower() in generic_locations_set]
    
    # Text Field Lengths (Average word/sentence counts from existing data)
    # Use median to be less sensitive to outliers in small dataset
    avg_user_bio_words = existing_df['User Bio'].astype(str).apply(lambda x: len(x.split())).median() if not existing_df['User Bio'].empty else 10
    avg_desc1_words = existing_df['User Description 1'].astype(str).apply(lambda x: len(x.split())).median() if not existing_df['User Description 1'].empty else 8
    avg_desc2_words = existing_df['User Description 2'].astype(str).apply(lambda x: len(x.split())).median() if not existing_df['User Description 2'].empty else 12
    avg_post_text_words = existing_df['Post Text'].astype(str).apply(lambda x: len(x.split())).median() if not existing_df['Post Text'].empty else 20

    # Ensure min lengths for Faker and add variance around median
    avg_user_bio_words = max(5, int(avg_user_bio_words))
    avg_desc1_words = max(5, int(avg_desc1_words))
    avg_desc2_words = max(5, int(avg_desc2_words))
    avg_post_text_words = max(10, int(avg_post_text_words))


    # --- 2. Generate synthetic data based on derived parameters ---
    for _ in range(num_rows):
        username = fake.user_name()
        # Use random.choices for categorical data to preserve distribution
        platform = random.choices(platforms, weights=platform_weights, k=1)[0]

        # Generate User Bio with length based on existing, and contact info probability
        user_bio_words = max(5, int(np.random.normal(avg_user_bio_words, 3))) # Add some variance
        user_bio = fake.sentence(nb_words=user_bio_words)
        if random.random() < prob_has_contact_info:
            user_bio += f" {fake.email()}" # Just an email example, could be phone/url also for variety.

        # Generate User Description 1 and 2 based on observed average lengths
        # Ensure they are always present, as original data indicates 100% content for desc1/desc2
        desc1_words = max(5, int(np.random.normal(avg_desc1_words, 2)))
        user_description1 = fake.sentence(nb_words=desc1_words)

        desc2_words = max(5, int(np.random.normal(avg_desc2_words, 3)))
        user_description2 = fake.sentence(nb_words=desc2_words)

        time_diff = max_date - min_date
        random_days = random.randint(0, max(1, time_diff.days))
        account_creation_date = (min_date + timedelta(days=random_days)).strftime('%d/%m/%Y')

        user_followers_raw = max(10, int(np.random.lognormal(mean=followers_mean_log, sigma=followers_std_log)))
        user_followers = str(user_followers_raw)

        # Engagement: Use derived ratios
        likes_reactions_raw = int(user_followers_raw * random.uniform(min_likes_ratio, max_likes_ratio))
        comments_raw = int(likes_reactions_raw * random.uniform(min_comments_ratio, max_comments_ratio) * (likes_reactions_raw if likes_reactions_raw > 0 else 1))
        shares_retweets_raw = int(likes_reactions_raw * random.uniform(min_shares_ratio, max_shares_ratio) * (likes_reactions_raw if likes_reactions_raw > 0 else 1))

        likes_reactions = str(max(0, likes_reactions_raw))
        comments = str(max(0, comments_raw))
        shares_retweets = str(max(0, shares_retweets_raw))

        # User Activity
        posts_count_week = random.randint(min_posts_per_week, max_posts_per_week)
        # Randomly choose if it's per week or per day, based on some assumed split
        if random.random() < 0.8: # 80% chance of 'per week' (common for most platforms)
            user_activity = f"{posts_count_week} posts per week"
        else: # 20% chance of 'per day'
            posts_count_day = max(1, round(posts_count_week / 7)) # Convert roughly to daily
            user_activity = f"{posts_count_day} posts per day"
            if random.random() < 0.1: # occasionally add 's' for singular 'post'
                user_activity = user_activity.replace('posts', 'post', 1)

        # Post Text
        post_text_words = max(10, int(np.random.normal(avg_post_text_words, 5)))
        post_text = fake.paragraph(nb_sentences=max(1, int(post_text_words / 15))) # Rough conversion to sentences
        sensitive_keywords_list = ['password', 'credit card', 'medical record', 'social security number', 'bank account', 'home address', 'phone number', 'my ssn', 'dob', 'private info'] # Expanded list
        if random.random() < prob_has_sensitive_keywords:
            post_text += f" {random.choice(sensitive_keywords_list)}"

        hashtags_count = random.randint(min_hashtags, max_hashtags)
        hashtags = ' '.join([f'#{fake.word()}' for _ in range(hashtags_count)])
        if hashtags_count == 0:
            hashtags = ''

        # Location generation based on derived proportions
        if random.random() < prob_is_specific_location:
            if specific_existing_locations and random.random() < 0.7: # Higher chance to pick existing specific
                location = random.choice(specific_existing_locations)
            else:
                location = fake.city() # New specific location
        else:
            if generic_existing_locations and random.random() < 0.7: # Higher chance to pick existing generic
                location = random.choice(generic_existing_locations)
            else:
                location = random.choice(list(generic_locations_set)) # New generic location

        media_type = random.choices(media_types, weights=media_weights, k=1)[0]
        privacy_setting = random.choices(privacy_settings_options, weights=privacy_weights, k=1)[0]
        account_verification = random.choices(account_verification_options, weights=verification_weights, k=1)[0]

        row = {
            'Username': username,
            'Platform': platform,
            'User Bio': user_bio,
            'User Description 1': user_description1,
            'User Description 2': user_description2,
            'Account Creation Date': account_creation_date,
            'User Followers': user_followers,
            'Likes/Reactions': likes_reactions,
            'Comments': comments,
            'Shares/Retweets': shares_retweets,
            'User Activity': user_activity,
            'Post Text': post_text,
            'Hashtags': hashtags,
            'Location': location,
            'Media Type': media_type,
            'Privacy Settings': privacy_setting,
            'Account Verification': account_verification
            # 'Language' field is intentionally excluded
        }
        data.append(row)

    return pd.DataFrame(data)

# --- Main execution for `generate_synthetic_data.py` ---
if __name__ == "__main__":
    original_input_file = 'socialmedia.csv'
    original_df = None
    try:
        original_df = pd.read_csv(original_input_file)
        print(f"Loaded original '{original_input_file}' with {len(original_df)} rows.")
        if len(original_df) < 5:
            print("Warning: Original dataset has very few rows. Derived parameters might not be robust.")
    except FileNotFoundError:
        print(f"Error: '{original_input_file}' not found. Cannot base synthetic data on existing data.")
        print("Please ensure 'socialmedia.csv' is in the same directory.")
        exit()

    num_synthetic_rows = 2000 # You can change this number
    print(f"Generating {num_synthetic_rows} synthetic social media data rows based on existing data...")
    synthetic_df = generate_synthetic_social_media_data_based_on_existing(original_df, num_synthetic_rows)
    print("Synthetic data generated.")

    extended_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    print(f"Combined dataset has {len(extended_df)} rows.")

    output_file_name = 'extended_socialmedia.csv'
    extended_df.to_csv(output_file_name, index=False)
    print(f"Extended dataset saved to '{output_file_name}'.")

    print("\nSample of the extended dataset:")
    print(extended_df.head())
    print("\nColumn Info:")
    extended_df.info()

```