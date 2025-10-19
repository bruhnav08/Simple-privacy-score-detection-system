# generate_synthetic_data.py

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import re

# Initialize Faker
fake = Faker()

# --- Helper Functions ---

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
    with distributions based on an existing DataFrame, including all original columns.
    This version ensures diversity for certain features even if original data was uniform,
    prioritizing relevance for formula and research.

    Args:
        existing_df (pandas.DataFrame): The DataFrame with existing data (your 40 rows) as a template.
        num_rows (int): The number of rows (users/posts) to generate.

    Returns:
        pandas.DataFrame: A DataFrame containing the synthetic data (only synthetic rows).
    """
    data = []

    # --- 1. Derive parameters from existing_df for improved accuracy ---

    # Helper for categorical distributions
    def get_categorical_distribution(series, default_options):
        cleaned_series = series.dropna().astype(str)
        if not cleaned_series.empty:
            counts = cleaned_series.value_counts(normalize=True)
            observed_options = list(counts.index)
            observed_weights = list(counts.values)
            return observed_options, observed_weights
        return default_options, [1/len(default_options)]*len(default_options) if default_options else ([],[])

    # Categorical Features
    platforms, platform_weights = get_categorical_distribution(existing_df['Platform'], ['Instagram', 'Twitter', 'Facebook', 'LinkedIn', 'TikTok', 'Threads', 'Reddit', 'Other'])
    media_types, media_weights = get_categorical_distribution(existing_df['Media Type'], ['video', 'image', 'text', 'unknown'])
    privacy_settings_options, privacy_weights = get_categorical_distribution(existing_df['Privacy Settings'], ['public', 'private', 'friends only', 'limited', 'followers only'])
    account_verification_options, verification_weights = get_categorical_distribution(existing_df['Account Verification'], ['verified', 'not verified', 'yes', 'no', 'true', 'false'])

    # Date Range
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

    # Numerical Features - User Followers
    try:
        existing_followers = existing_df['User Followers'].astype(str).str.replace(',', '').astype(float).dropna()
        if not existing_followers.empty and existing_followers.min() > 0:
            log_followers = np.log(existing_followers[existing_followers > 0])
            followers_mean_log = log_followers.mean()
            followers_std_log = log_followers.std()
            if np.isnan(followers_std_log) or followers_std_log == 0: followers_std_log = 0.5
            if np.isnan(followers_mean_log): followers_mean_log = np.log(1000)
        else:
            followers_mean_log = np.log(1000); followers_std_log = 2.0
    except Exception:
        followers_mean_log = np.log(1000); followers_std_log = 2.0

    # Numerical Features - User Following (similar to followers, but often smaller)
    try:
        existing_following = existing_df['User Following'].astype(float).dropna()
        if not existing_following.empty and existing_following.min() > 0:
            log_following = np.log(existing_following[existing_following > 0])
            following_mean_log = log_following.mean()
            following_std_log = log_following.std()
            if np.isnan(following_std_log) or following_std_log == 0: following_std_log = 0.5
            if np.isnan(following_mean_log): following_mean_log = np.log(400)
        else:
            following_mean_log = np.log(400); following_std_log = 1.0
    except Exception:
        following_mean_log = np.log(400); following_std_log = 1.0

    # Engagement: Likes/Comments/Shares - Derive typical engagement ratios from existing data
    try:
        existing_likes = existing_df['Likes/Reactions'].astype(str).str.replace(',', '').astype(float).dropna()
        existing_comments = existing_df['Comments'].astype(str).str.replace(',', '').astype(float).dropna()
        existing_shares = existing_df['Shares/Retweets'].astype(str).str.replace(',', '').astype(float).dropna()
        
        temp_df_eng = pd.DataFrame({
            'followers': existing_followers,
            'likes': existing_likes,
            'comments': existing_comments,
            'shares': existing_shares
        }).dropna()

        avg_likes_per_follower = (temp_df_eng['likes'] / temp_df_eng['followers']).replace([np.inf, -np.inf], np.nan).dropna().mean()
        avg_comments_per_like = (temp_df_eng['comments'] / temp_df_eng['likes']).replace([np.inf, -np.inf], np.nan).dropna().mean()
        avg_shares_per_like = (temp_df_eng['shares'] / temp_df_eng['likes']).replace([np.inf, -np.inf], np.nan).dropna().mean()
        
        if np.isnan(avg_likes_per_follower): avg_likes_per_follower = 0.11
        if np.isnan(avg_comments_per_like): avg_comments_per_like = 0.1
        if np.isnan(avg_shares_per_like): avg_shares_per_like = 0.05

        min_likes_ratio = max(0.001, avg_likes_per_follower * 0.8)
        max_likes_ratio = avg_likes_per_follower * 1.2
        min_comments_ratio = max(0.01, avg_comments_per_like * 0.8)
        max_comments_ratio = avg_comments_per_like * 1.2
        min_shares_ratio = max(0.005, avg_shares_per_like * 0.8)
        max_shares_ratio = avg_shares_per_like * 1.2

        max_likes_ratio = min(max_likes_ratio, 0.5)
        max_comments_ratio = min(max_comments_ratio, 0.4)
        max_shares_ratio = min(max_shares_ratio, 0.2)

    except Exception as e:
        print(f"Error deriving engagement ratios from existing data: {e}. Using robust defaults.")
        min_likes_ratio, max_likes_ratio = 0.05, 0.12
        min_comments_ratio, max_comments_ratio = 0.08, 0.18
        min_shares_ratio, max_shares_ratio = 0.04, 0.1

    # User Engagement (specific field from data)
    try:
        existing_user_engagement = existing_df['User Engagement'].astype(float).dropna()
        if not existing_user_engagement.empty:
            user_engagement_mean = existing_user_engagement.mean()
            user_engagement_std = existing_user_engagement.std()
            if np.isnan(user_engagement_std) or user_engagement_std == 0: user_engagement_std = user_engagement_mean * 0.15
        else:
            user_engagement_mean = 1200; user_engagement_std = 250
    except Exception:
        user_engagement_mean = 1200; user_engagement_std = 250

    # User Interactions (specific field from data)
    try:
        existing_user_interactions = existing_df['User Interactions'].astype(float).dropna()
        if not existing_user_interactions.empty:
            user_interactions_mean = existing_user_interactions.mean()
            user_interactions_std = existing_user_interactions.std()
            if np.isnan(user_interactions_std) or user_interactions_std == 0: user_interactions_std = user_interactions_mean * 0.15
        else:
            user_interactions_mean = 11000; user_interactions_std = 2500
    except Exception:
        user_interactions_mean = 11000; user_interactions_std = 2500


    # Posts per week: Match observed min/max and mean more closely
    try:
        existing_posts_per_week = existing_df['User Activity'].apply(extract_posts_per_week).dropna()
        if not existing_posts_per_week.empty:
            posts_per_week_mean_target = existing_posts_per_week.mean()
            posts_per_week_std_target = existing_posts_per_week.std()
            if np.isnan(posts_per_week_std_target) or posts_per_week_std_target == 0: posts_per_week_std_target = 1.0
            
            min_posts_per_week = max(1, int(posts_per_week_mean_target - posts_per_week_std_target * 1.5))
            max_posts_per_week = int(posts_per_week_mean_target + posts_per_week_std_target * 1.5)
            max_posts_per_week = max(min_posts_per_week + 1, max_posts_per_week)
        else:
            min_posts_per_week, max_posts_per_week = 1, 7 # Adjusted default range
    except Exception:
        min_posts_per_week, max_posts_per_week = 1, 7

    # Hashtags count range: CRITICAL FIX - Original was very tight (0-1, mean 0.975)
    try:
        existing_hashtag_counts = existing_df['Hashtags'].astype(str).apply(lambda x: len(re.findall(r'#\w+', x))).dropna()
        if not existing_hashtag_counts.empty:
            # If original data is almost always 1, make synthetic similar but allow for 0-2
            if existing_hashtag_counts.nunique() == 1 and existing_hashtag_counts.iloc[0] == 1:
                min_hashtags, max_hashtags = 0, 2 # Allow 0-2 for variance, but weighted towards 1
            else:
                min_hashtags = int(existing_hashtag_counts.min())
                max_hashtags = int(existing_hashtag_counts.max())
                min_hashtags = max(0, min_hashtags - 1)
                max_hashtags = max_hashtags + 1
            if min_hashtags > max_hashtags: min_hashtags, max_hashtags = max_hashtags, min_hashtags
        else:
            min_hashtags, max_hashtags = 0, 2 # Default very tight
    except Exception:
        min_hashtags, max_hashtags = 0, 2

    # Binary Feature Proportions (introducing variance where original was uniform)
    # Default to non-zero values if original was 0, or non-one if original was 1.
    prob_has_contact_info = existing_df['User Bio'].apply(has_contact_info).mean() if not existing_df.empty else 0.0
    if prob_has_contact_info == 0.0: prob_has_contact_info = 0.05 # Inject 5% chance if original had none

    prob_has_sensitive_keywords = existing_df['Post Text'].apply(has_sensitive_keywords).mean() if not existing_df.empty else 0.05
    # If original was very low, ensure it's still low but allows some instances for learning
    if prob_has_sensitive_keywords < 0.05: prob_has_sensitive_keywords = 0.08 # Ensure at least 8% if very low/none

    prob_is_specific_location = existing_df['Location'].apply(is_specific_location).mean() if not existing_df.empty else 0.95
    if prob_is_specific_location == 1.0: prob_is_specific_location = 0.90 # Allow 10% generic if original was all specific


    # Location: Derived from existing data
    generic_locations_set = {'unknown', 'global', 'worldwide', 'earth', 'planet', 'everywhere', 'anywhere', 'all', 'none', 'n/a', 'na', ''}
    existing_locations_list = existing_df['Location'].dropna().unique().tolist()
    specific_existing_locations = [loc for loc in existing_locations_list if str(loc).strip().lower() not in generic_locations_set]
    generic_existing_locations = [loc for loc in existing_locations_list if str(loc).strip().lower() in generic_locations_set]
    
    # Server Post: Unique values from existing data
    server_posts_options, server_posts_weights = get_categorical_distribution(existing_df['Server Post'], ['New York City, USA', 'London, UK', 'Los Angeles, USA', 'Paris, France', 'Tokyo, Japan'])

    # Text Field Lengths (Average word/sentence counts from existing data)
    avg_user_bio_words = existing_df['User Bio'].astype(str).apply(lambda x: len(x.split())).median() if not existing_df['User Bio'].empty else 10
    avg_desc1_words = existing_df['User Description 1'].astype(str).apply(lambda x: len(x.split())).median() if not existing_df['User Description 1'].empty else 8
    avg_desc2_words = existing_df['User Description 2'].astype(str).apply(lambda x: len(x.split())).median() if not existing_df['User Description 2'].empty else 12
    avg_post_text_words = existing_df['Post Text'].astype(str).apply(lambda x: len(x.split())).median() if not existing_df['Post Text'].empty else 20

    # Adjusted STDs for more variance in word counts, but still based on original medians
    avg_user_bio_words = max(5, int(avg_user_bio_words))
    avg_desc1_words = max(5, int(avg_desc1_words))
    avg_desc2_words = max(5, int(avg_desc2_words))
    avg_post_text_words = max(10, int(avg_post_text_words))
    
    # Derive Media URL None probability from existing data. Adjust slightly to ensure variance.
    prob_media_url_none = existing_df['Media URL'].isnull().mean() if 'Media URL' in existing_df.columns and not existing_df['Media URL'].empty else 0.35
    prob_media_url_none = max(0.2, min(0.8, prob_media_url_none)) # Cap between 20-80% to ensure both pres/abs for learning


    # Derive Mentions content probability (inject very small non-zero probability if original had none)
    prob_mentions_has_content = 0.0
    if 'Mentions' in existing_df.columns:
        observed_mentions_prop = (existing_df['Mentions'].astype(str).apply(lambda x: len(x.strip()) > 0)).mean()
        if observed_mentions_prop > 0: # If original had some mentions, use that prop (but cap it low)
            prob_mentions_has_content = min(0.05, observed_mentions_prop) # Cap at 5% to avoid too many
        else: # If original had NO mentions (prop is 0), inject a very small chance for variance
            prob_mentions_has_content = 0.005 # Inject 0.5% chance if original had none


    # --- 2. Generate synthetic data based on derived parameters ---
    for i in range(num_rows):
        # Select platform and media_type first for contextual generation
        platform = random.choices(platforms, weights=platform_weights, k=1)[0]
        media_type = random.choices(media_types, weights=media_weights, k=1)[0]

        username = fake.user_name()

        # User ID: Generate unique IDs within the observed range (or wider if needed for more diversity)
        user_id = random.randint(100, 1000) # Simple range, ensures variety

        # Post ID: Generate unique IDs within the observed range
        post_id = random.randint(10000, 99999)

        # Generate User Bio with length based on existing, and contact info probability
        user_bio_words = max(5, int(np.random.normal(avg_user_bio_words, 3))) # Slightly wider STD
        user_bio = fake.sentence(nb_words=user_bio_words)
        if random.random() < prob_has_contact_info:
            user_bio += f" Contact: {fake.email()}"

        desc1_words = max(5, int(np.random.normal(avg_desc1_words, 2))) # Slightly wider STD
        user_description1 = fake.sentence(nb_words=desc1_words)

        desc2_words = max(5, int(np.random.normal(avg_desc2_words, 3))) # Slightly wider STD
        user_description2 = fake.sentence(nb_words=desc2_words)

        # Post Timestamp: Generate a timestamp after account creation date and before current date
        time_diff_creation = max_date - min_date
        random_days_creation = random.randint(0, max(1, time_diff_creation.days))
        account_creation_date_val = min_date + timedelta(days=random_days_creation)
        account_creation_date = account_creation_date_val.strftime('%d/%m/%Y')

        post_timestamp_date_val = account_creation_date_val + timedelta(days=random.randint(0, (CURRENT_DATE - account_creation_date_val).days))
        post_timestamp_date = post_timestamp_date_val.strftime('%Y-%m-%d %H:%M:%S')


        user_followers_raw = max(10, int(np.random.lognormal(mean=followers_mean_log, sigma=followers_std_log)))
        user_followers = str(user_followers_raw)

        # User Following
        user_following_raw = max(10, int(np.random.lognormal(mean=following_mean_log, sigma=following_std_log)))
        user_following = str(user_following_raw)

        # Engagement: Use derived ratios
        likes_reactions_raw = int(user_followers_raw * random.uniform(min_likes_ratio, max_likes_ratio))
        comments_raw = int(likes_reactions_raw * random.uniform(min_comments_ratio, max_comments_ratio))
        shares_retweets_raw = int(likes_reactions_raw * random.uniform(min_shares_ratio, max_shares_ratio))

        likes_reactions = str(max(0, likes_reactions_raw))
        comments = str(max(0, comments_raw))
        shares_retweets = str(max(0, shares_retweets_raw))
        
        # User Engagement (the specific field in original data)
        user_engagement_val = max(0, int(np.random.normal(user_engagement_mean, user_engagement_std)))
        user_engagement = str(user_engagement_val)

        # User Interactions (the specific field in original data)
        user_interactions_val = max(0, int(np.random.normal(user_interactions_mean, user_interactions_std)))
        user_interactions = str(user_interactions_val)

        # User Activity
        posts_count_week = max(1, int(np.random.normal(posts_per_week_mean_target, posts_per_week_std_target)))
        posts_count_week = min(max_posts_per_week, max(min_posts_per_week, posts_count_week))

        if random.random() < 0.8:
            user_activity = f"{posts_count_week} posts per week"
        else:
            posts_count_day = max(1, round(posts_count_week / 7))
            user_activity = f"{posts_count_day} posts per day"
            if random.random() < 0.1:
                user_activity = user_activity.replace('posts', 'post', 1)

        # Post Text
        post_text_words = max(10, int(np.random.normal(avg_post_text_words, 5)))
        post_text = fake.paragraph(nb_sentences=max(1, int(post_text_words / 15)))
        sensitive_keywords_list = ['password', 'credit card', 'medical record', 'social security number', 'bank account', 'home address', 'phone number', 'my ssn', 'dob', 'private info']
        if random.random() < prob_has_sensitive_keywords:
            post_text += f" {random.choice(sensitive_keywords_list)}"

        # Hashtags count: Generate based on (1, 0.5) to ensure 0-2 range with peak at 1
        hashtags_count = max(min_hashtags, min(max_hashtags, round(np.random.normal(1, 0.5))))
        hashtags_list = [f'#{fake.word()}' for _ in range(hashtags_count)]
        hashtags = ' '.join(hashtags_list)

        # Mentions: Now uses prob_mentions_has_content
        mentions = ''
        if random.random() < prob_mentions_has_content:
            mentions = f"@{fake.user_name()}"


        # Media URL
        media_url = None
        if random.random() > prob_media_url_none: # Controls proportion of None values
            if media_type.lower() == 'image':
                media_url = fake.image_url()
            elif media_type.lower() == 'video':
                media_url = fake.url() + '.mp4'
            elif media_type.lower() == 'text':
                media_url = None
            else: # unknown
                media_url = fake.url() if random.random() < 0.5 else None


        # Post URL
        post_url_base = ""
        if platform.lower() == 'instagram':
            post_url_base = f"http://instagram.com/{username}/"
        elif platform.lower() == 'twitter':
            post_url_base = f"http://twitter.com/{username}/"
        elif platform.lower() == 'facebook':
            post_url_base = f"http://facebook.com/{username}/posts/"
        elif platform.lower() == 'linkedin':
            post_url_base = f"http://linkedin.com/posts/{username}/"
        elif platform.lower() == 'tiktok':
            post_url_base = f"http://tiktok.com/@{username}/video/"
        elif platform.lower() == 'reddit':
            post_url_base = f"http://reddit.com/r/{fake.word()}/comments/{fake.hexify(text='^^^', upper=False)}/"
        else:
            post_url_base = fake.url() + '/post/'
        
        if platform.lower() in ['instagram', 'twitter', 'tiktok', 'linkedin']:
            post_url = post_url_base + str(post_id)
        elif platform.lower() == 'facebook':
            post_url = f"{post_url_base}{fake.hexify(text='^^^^^^^^^^^^^^^^^^^^', upper=False)}/"
        elif platform.lower() == 'reddit':
            post_url = post_url_base + fake.slug()
        else:
            post_url = post_url_base + str(post_id)


        # Location generation based on derived proportions
        if random.random() < prob_is_specific_location:
            if specific_existing_locations and random.random() < 0.8:
                location = random.choice(specific_existing_locations)
            else:
                location = fake.city()
        else:
            if generic_existing_locations and random.random() < 0.8:
                location = random.choice(generic_existing_locations)
            else:
                location = random.choice(list(generic_locations_set))

        media_type = random.choices(media_types, weights=media_weights, k=1)[0]
        privacy_setting = random.choices(privacy_settings_options, weights=privacy_weights, k=1)[0]
        account_verification = random.choices(account_verification_options, weights=verification_weights, k=1)[0]

        # Server Post
        server_post = random.choices(server_posts_options, weights=server_posts_weights, k=1)[0]

        row = {
            'User ID': user_id,
            'Username': username,
            'Platform': platform,
            'Post ID': post_id,
            'Post Text': post_text,
            'Post Timestamp': post_timestamp_date,
            'Likes/Reactions': likes_reactions,
            'Comments': comments,
            'Shares/Retweets': shares_retweets,
            'Hashtags': hashtags,
            'Mentions': mentions,
            'Media Type': media_type,
            'Media URL': media_url,
            'Post URL': post_url,
            'Location': location,
            'Privacy Settings': privacy_setting,
            'User Followers': user_followers,
            'User Following': user_following,
            'Account Creation Date': account_creation_date,
            'Account Verification': account_verification,
            'User Engagement': user_engagement,
            'User Interactions': user_interactions,
            'User Activity': user_activity,
            'User Bio': user_bio,
            'User Description 1': user_description1,
            'User Description 2': user_description2,
            'Server Post': server_post
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

    num_synthetic_rows = 2000
    print(f"Generating {num_synthetic_rows} synthetic social media data rows based on existing data...")
    synthetic_df = generate_synthetic_social_media_data_based_on_existing(original_df, num_synthetic_rows)
    print("Synthetic data generated.")

    extended_df = synthetic_df # The extended_df now ONLY contains the synthetic data
    print(f"Extended dataset contains {len(extended_df)} synthetic rows (original 40 rows used as template only).")

    output_file_name = 'extended_socialmedia.csv'
    extended_df.to_csv(output_file_name, index=False)
    print(f"Extended dataset saved to '{output_file_name}'.")

    print("\nSample of the extended dataset:")
    print(extended_df.head())
    print("\nColumn Info:")
    extended_df.info()