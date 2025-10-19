# Social Media Privacy Score Analyzer

This project uses a machine learning model to analyze social media profile data and calculate a "Comprehensive Privacy Score" for each user. It identifies key risk factors by processing raw user data, engineering dozens of features (like sentiment, post frequency, and keyword density), and training a model to quantify privacy exposure.

## Core Concept

The goal is to create a nuanced privacy score that goes beyond simple checks. The workflow is:

1.  **Generate Data:** A small seed dataset (`socialmedia.csv`) is used by `generate_synthetic_data.py` to create a large, realistic dataset (`extended_socialmedia.csv`).
2.  **Engineer Features:** The `1.py` script cleans the data and performs heavy feature engineering. It extracts ~30 new features, including sentiment scores, account age, text-to-image ratios, and keyword analysis, saving the result as `preprocessed_socialmedia.csv`.
3.  **Define a Target:** A "proxy risk score" is created based on clear-cut high-risk features (e.g., has public contact info, uses specific locations, has sensitive keywords). This score acts as the "ground truth" for the model to learn from.
4.  **Train & Predict:** `direct_privacy_scorer.py` trains a Random Forest Regressor to learn the complex relationships between *all* 30+ features and the "proxy risk score."
5.  **Generate Final Score:** The trained model is then used to predict a final **Comprehensive Privacy Score** for every user. This score is more robust because it's learned from all features, not just the obvious ones.
6.  **Visualize:** `visualize_scores.py` plots the distribution of scores and the most important features that contribute to a high or low score.

## Project Workflow

![A diagram showing the project workflow: extended_socialmedia.csv goes into 1.py, which creates preprocessed_socialmedia.csv. This goes into direct_privacy_scorer.py, which outputs users_with_comprehensive_privacy_scores.csv and direct_ml_feature_importances.csv. These two files, along with preprocessed_socialmedia.csv, go into visualize_scores.py, which creates the final plots.](https://i.imgur.com/gA3q3kE.png)

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    * You should create a `requirements.txt` file. Based on your scripts, it would look like this:
        ```
        pandas
        numpy
        scikit-learn
        textblob
        nltk
        matplotlib
        seaborn
        faker
        ```
    * Run the installation:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Download NLTK Data:**
    The `1.py` script attempts to download the NLTK `stopwords` and `punkt` packages automatically. If this fails, you can run this command in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

## How to Run the Project

The scripts are designed to be run in a specific order.

1.  **(Optional) Generate Synthetic Data:**
    If you want to re-generate the `extended_socialmedia.csv` file:
    ```bash
    python generate_synthetic_data.py
    ```

2.  **Preprocess Data & Engineer Features:**
    This script reads `extended_socialmedia.csv` and creates `preprocessed_socialmedia.csv`.
    ```bash
    python 1.py
    ```

3.  **Calculate Privacy Scores & Feature Importances:**
    This script reads `preprocessed_socialmedia.csv` and creates two new files:
    * `users_with_comprehensive_privacy_scores.csv`
    * `direct_ml_feature_importances.csv`
    ```bash
    python direct_privacy_scorer.py
    ```

4.  **Visualize the Results:**
    This script reads the output files from the previous steps to generate and display plots.
    ```bash
    python visualize_scores.py
    ```

## File Descriptions

### Key Python Scripts (`.py`)

* `generate_synthetic_data.py`: Uses `faker` to create a large, synthetic dataset (`extended_socialmedia.csv`) based on the schema of `socialmedia.csv`.
* `1.py`: The main preprocessing script. It loads `extended_socialmedia.csv`, cleans it, and engineers ~30 new features (sentiment, normalization, keyword analysis, etc.), saving the result as `preprocessed_socialmedia.csv`.
* `direct_privacy_scorer.py`: The core ML script. It loads the preprocessed data, defines a target "proxy risk score," trains a `RandomForestRegressor`, and uses it to predict the final `comprehensive_privacy_score` for all users.
* `visualize_scores.py`: Reads all the final CSV files to create a dashboard of plots showing score distributions and feature importances.
* `data_snapshot_analyser.py`: A helper utility to print the `head()` and `info()` of the different CSV files.
* `derive_weights.py`: An exploratory script showing an alternative method for generating feature importances.

### Key Data Files (`.csv`)

* `socialmedia.csv`: The original, small (40-row) seed dataset.
* `extended_socialmedia.csv`: The large (2000-row) synthetic dataset generated from the script. **This is the main input data.**
* `preprocessed_socialmedia.csv`: The fully cleaned, normalized, and feature-engineered dataset. **This is the main input for the ML model.**
* `users_with_comprehensive_privacy_scores.csv`: **Final Output.** A two-column CSV (`User ID`, `comprehensive_privacy_score`) containing the final score for each user.
* `direct_ml_feature_importances.csv`: **Final Output.** A two-column CSV (`Feature`, `Importance`) showing which features contributed most to the model's predictions.

---

**Note for your repository:** You should add `tempCodeRunnerFile.py` and any `.xlsx` files to a `.gitignore` file so they aren't tracked by Git.
