# Comment Category Prediction

> **Multi-class NLP classification** using TF-IDF feature engineering and ensemble ML on structured + textual comment data.

---

## Problem Statement

Given a dataset of user comments along with metadata such as upvotes, downvotes, emoticons, user demographics (race, religion, gender), and post information, the task is to **predict the category (label) assigned to each comment**. This is a multi-class text classification problem.

The dataset contains:
- `comment` — raw text of the user comment
- `upvote`, `downvote` — community engagement metrics
- `emoticon_1`, `emoticon_2`, `emoticon_3` — emoticon usage counts
- `race`, `religion`, `gender` — demographic attributes
- `post_id`, `created_date` — post and time metadata
- `label` — target variable (multi-class category)

---

## Approach & Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed label distribution to check for class imbalance
- Generated correlation heatmaps for numerical features
- Visualized comment length distribution, upvote/downvote patterns, and word counts per class
- Extracted temporal patterns (comments by hour, day of week)
- Analyzed exclamation counts per class to identify linguistic patterns

### 2. Text Preprocessing
Custom `clean_text()` function applied to all comments:
- Lowercasing
- URL replacement with `url` token
- Special character removal
- Whitespace normalization

### 3. Handling Missing Values & Encoding
- Replaced `"none"` values with `"unknown"` in categorical columns
- Filled NaNs in `race`, `religion`, `gender` with `"missing"`
- Applied **Label Encoding** consistently across train and test sets (fit on combined data to avoid unseen-label errors)

### 4. Feature Engineering
A rich set of hand-crafted features was created from the raw data:

**Text Features:**
| Feature | Description |
|---|---|
| `comment_length` | Total character count |
| `word_count` | Number of words |
| `avg_word_length` | Average characters per word |
| `unique_word_ratio` | Lexical diversity |
| `char_per_word` | Compactness of writing |
| `exclamation_count` | Emotional intensity signals |
| `question_count` | Interrogative patterns |
| `caps_ratio` | Capitalization as a tone indicator |

**Engagement Features:**
| Feature | Description |
|---|---|
| `total_votes` | Sum of upvotes and downvotes |
| `vote_ratio` | Upvote / (downvote + 1) |
| `vote_diff` | Net community sentiment |
| `log_upvote`, `log_downvote` | Log-transformed engagement |

**Emoticon Features:**
| Feature | Description |
|---|---|
| `total_emoticons` | Sum of all emoticon columns |
| `emoticon_ratio` | Emoticon type ratio |
| `has_emoticon` | Binary flag |

**Post-Level Aggregated Features:**
- `post_size` — number of comments per post
- `post_avg_upvote`, `post_avg_downvote` — community engagement norms
- `post_avg_length` — typical comment length for that post

**Temporal Features:**
- `hour`, `dayofweek`, `month`, `is_weekend`

### 5. TF-IDF Vectorization (Dual-Level)
Two TF-IDF representations were extracted from the cleaned comment text:

| Vectorizer | Config |
|---|---|
| Word-level TF-IDF | `max_features=30000`, `ngram_range=(1,3)`, `sublinear_tf=True`, `min_df=3` |
| Char-level TF-IDF | `max_features=30000`, `ngram_range=(2,4)`, `analyzer='char_wb'`, `min_df=5` |

Both were combined with the numerical features into a single sparse matrix using `scipy.sparse.hstack`.

### 6. Model Training & Comparison
Four classifiers were trained and compared on a stratified 90/10 train-val split:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| LinearSVC | Efficient SVM for sparse text data |
| SGDClassifier | Stochastic gradient descent |
| **LightGBM** | Gradient boosting — selected as best model |

Evaluation metrics: **Accuracy, Precision, Recall, F1-Score (weighted)**

Confusion matrix and per-class TP/TN/FP/FN breakdown were generated for full transparency.

### 7. Final Model & Submission
The final LightGBM model was retrained on the full training set with tuned hyperparameters:
- `n_estimators=2000`, `learning_rate=0.03`, `num_leaves=96`, `max_depth=8`
- `class_weight="balanced"` to handle any class imbalance
- `reg_alpha=0.1`, `reg_lambda=1.0` for regularization

Predictions were generated for the test set and submitted as `submission.csv`.

---

## Tech Stack

- **Language:** Python 3
- **Core Libraries:** pandas, numpy, scikit-learn, LightGBM, scipy
- **NLP:** TF-IDF (word + char level), regex-based text cleaning
- **Visualization:** matplotlib, seaborn
- **Platform:** Collab Notebook

---

## Key Takeaways

- Combining **word-level and character-level TF-IDF** captures both semantic meaning and morphological patterns, improving model robustness
- **Hand-crafted engagement and stylistic features** (vote ratios, caps ratio, emoticon counts) provided complementary signal beyond raw text
- **Post-level aggregation** contextualized individual comments within their discussion thread
- **LightGBM** outperformed linear classifiers (LR, SVM, SGD) on this mixed feature space
- **`class_weight="balanced"`** on the final model helped handle any label skew in the training data

---
