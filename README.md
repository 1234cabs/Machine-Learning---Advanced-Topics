# Machine Learning Advanced Topics

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-Latest-green.svg)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/tutkufurkan/Machine-Learning---Advanced-Topics)

## Overview

A comprehensive tutorial covering **advanced machine learning techniques** including Natural Language Processing (NLP), dimensionality reduction with PCA, hyperparameter optimization, and collaborative filtering recommendation systems. Learn text classification, gender prediction from Twitter bios, optimal model selection, and movie recommendations using real-world datasets.

## 🎮 Interactive Demo

**👉 [Run the Interactive Notebook on Kaggle](https://www.kaggle.com/code/dandrandandran2093/machine-learning-advanced-topics)**

*Experience the full tutorial with pre-configured datasets, interactive visualizations, and word clouds on Kaggle!*

## Table of Contents

- [Introduction](#introduction)
- [Topics Covered](#topics-covered)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Performance Results](#performance-results)
- [Mathematical Foundations](#mathematical-foundations)
- [Contributing](#contributing)
- [References](#references)

## Introduction

This tutorial explores four essential advanced machine learning topics that bridge fundamental concepts with real-world applications. From understanding human language to reducing high-dimensional data, optimizing model performance, and building recommendation engines—these techniques represent the practical cutting edge of modern data science.

**What Makes This Advanced?**
- **Natural Language Processing**: Complete text preprocessing pipeline with gender classification
- **Dimensionality Reduction**: PCA for visualization while preserving 97% variance
- **Model Selection**: K-Fold Cross-Validation and Grid Search for optimal hyperparameters
- **Recommendation Systems**: Collaborative filtering with MovieLens dataset

## Topics Covered

### 📝 Part 1: Natural Language Processing (NLP)

**Goal:** Predict gender from Twitter bio text (~68% accuracy)

**Techniques Covered:**

#### 1. Regular Expression (RE)
- **Purpose**: Remove special characters, numbers, URLs, emojis
- **Pattern**: `[^a-zA-Z]` keeps only English letters
- **Result**: Clean, standardized text

**Example:**
```python
Original: "I ❤️ coding! Check my blog: https://example.com"
After RE: "I love coding Check my blog https example com"
```

#### 2. Stop Words Removal
- **Definition**: Common words with little semantic meaning
- **Examples**: "the", "is", "and", "of", "a"
- **Impact**: Reduces noise, focuses on meaningful words

**Why Remove?**
- "the cat is on the mat" → "cat mat" (kept meaningful words)
- Reduces feature space by ~40%
- Improves classification accuracy

#### 3. Lemmatization
- **Purpose**: Convert words to base/dictionary form
- **Examples**: 
  - "running" → "run"
  - "better" → "good"
  - "coding" → "code"
- **vs Stemming**: Lemmatization produces real words (better for readability)

#### 4. Data Cleaning Pipeline
Complete automated workflow:
```
Raw Text → Regex → Lowercase → Tokenization → Stop Words → Lemmatization → Clean Text
```

#### 5. Bag of Words (BoW)
- **Converts text to numerical vectors** for machine learning
- Creates sparse matrix: documents × vocabulary
- Each cell = word frequency in document
- **Parameters**: 
  - max_features=2000 (keep top 2000 words)
  - Sparsity: ~99% (most cells are zero)

**Example:**
```
Doc1: "cat sat mat"  → [1, 1, 1, 0, 0]
Doc2: "dog sat mat"  → [0, 1, 1, 1, 0]
Vocab: [cat, sat, mat, dog, ...]
```

#### 6. Text Classification
- **Algorithm**: Multinomial Naive Bayes
- **Why?** Fast, effective for text, handles high dimensionality
- **Features**: Bag of Words (2000 words)
- **Output**: Gender prediction (Male/Female)

#### 7. Word Clouds
- **Visualization**: Most frequent words by gender
- **Insight**: Reveals gender-specific language patterns
- **Male bios**: Tech, sports, business terms
- **Female bios**: Design, lifestyle, creative terms

**Applications:**
- Spam detection
- Sentiment analysis
- Customer feedback classification
- Social media analysis

---

### 📊 Part 2: Principal Component Analysis (PCA)

**Goal:** Reduce Iris dataset from 4D to 2D while keeping 97% variance

**What is PCA?**
- Unsupervised dimensionality reduction technique
- Finds directions (principal components) of maximum variance
- Projects data onto these components

**Key Concepts:**

#### Explained Variance
- **PC1**: ~73% of total variance (most important)
- **PC2**: ~23% of total variance
- **Total**: 97% variance preserved with just 2 dimensions!
- **Lost**: Only 3% information discarded

**Formula:**
```
Total Variance Explained = Σ(explained_variance_ratio)
PC1 + PC2 = 0.73 + 0.23 = 0.97 (97%)
```

#### Why Use PCA?

**Benefits:**
- **Visualization**: Can't plot 4D data, but 2D/3D is easy
- **Speed**: Fewer features → faster training
- **Overfitting**: Reduces model complexity
- **Storage**: Compressed representation

**When to Use:**
- High-dimensional data (many features)
- Correlated features (redundant information)
- Need visualization
- Training is slow

**When NOT to Use:**
- Features are already few and interpretable
- Need to understand individual feature impact
- Data is sparse (e.g., text after BoW)

#### 2D Visualization
- **3 Iris classes** clearly separated in 2D space
- Setosa (red) well-separated
- Versicolor (green) and Virginica (blue) slightly overlap
- Shows PCA preserves class structure

**Mathematical Steps:**
1. Standardize features (mean=0, std=1)
2. Compute covariance matrix
3. Calculate eigenvalues and eigenvectors
4. Select top K eigenvectors (K=2)
5. Transform data: `X_new = X × W`

**Whitening:**
- Scales components to unit variance
- Makes features equally important
- Useful before some algorithms (e.g., SVM)

---

### 🎯 Part 3: Model Selection

**Goal:** Find optimal hyperparameters using systematic search

#### 1. K-Fold Cross-Validation

**Problem with Single Split:**
- Train-test split depends on random seed
- One unlucky split → misleading results
- Not reliable for model comparison

**K-Fold Solution (K=10):**
```
Fold 1: [Test][Train][Train][Train][Train][Train][Train][Train][Train][Train]
Fold 2: [Train][Test][Train][Train][Train][Train][Train][Train][Train][Train]
...
Fold 10: [Train][Train][Train][Train][Train][Train][Train][Train][Train][Test]
```

**Process:**
1. Split data into K folds (K=10)
2. Train on K-1 folds (90% data)
3. Test on remaining fold (10% data)
4. Repeat K times, each fold is test once
5. Average results

**Advantages:**
- ✅ Every sample used for both training and testing
- ✅ Variance estimate (std of K scores)
- ✅ More reliable than single split
- ✅ No data wasted

**Results:**
- Mean accuracy across 10 folds
- Standard deviation (measures stability)
- Confidence in model performance

#### 2. Grid Search with KNN

**Problem:** How to choose K?
- K=1: Overfitting (memorizes training data)
- K=50: Underfitting (too smooth, loses patterns)
- Need optimal K!

**Grid Search Solution:**
- Try K = 1, 2, 3, ..., 50
- Use 10-fold CV for each K
- Choose K with highest average accuracy

**Process:**
```python
grid = {"n_neighbors": [1, 2, 3, ..., 50]}
GridSearchCV → Try all values → Find best
```

**Results:**
- **Best K found** (e.g., K=13)
- Plot: K vs Accuracy curve
- Shows overfitting (low K) and underfitting (high K)

**Visualization:**
- X-axis: K values
- Y-axis: Accuracy
- Red line: Optimal K
- Shows trade-off clearly

#### 3. Grid Search with Logistic Regression

**Hyperparameters to Tune:**
- **C (Regularization)**: Controls overfitting
  - Small C → Strong regularization → Simple model
  - Large C → Weak regularization → Complex model
- **Penalty**: L1 (Lasso) or L2 (Ridge)
  - L1: Feature selection (sparse weights)
  - L2: Weight shrinkage (all features used)

**Search Space:**
```python
C: [0.001, 0.01, 0.1, 1, 10, 100, 1000]  # log scale
penalty: ['l1', 'l2']
Total combinations: 7 × 2 = 14
```

**Process:**
1. Try all 14 combinations
2. Use 10-fold CV for each
3. Find best C and penalty
4. Retrain on full data with best params
5. Evaluate on test set

**Results:**
- Best C value (e.g., C=1)
- Best penalty (e.g., L2)
- Test accuracy with optimal settings

**Why This Matters:**
- Systematic vs random tuning
- Reproducible results
- Confidence in model selection
- Essential for competitions/production

---

### 🎬 Part 4: Recommendation Systems

**Goal:** Recommend movies similar to "Bad Boys (1995)"

#### What are Recommendation Systems?

**Definition:** Predict user preferences and suggest relevant items

**Real-World Examples:**
- Netflix: Movie/series recommendations
- Amazon: Product suggestions
- Spotify: Music discovery
- YouTube: Video recommendations

**Two Main Approaches:**

#### Content-Based Filtering
- Recommends items **similar to what user liked**
- Uses item features (genre, actors, director)
- Example: You liked Action → Recommend Action movies

**Advantages:**
- No cold start for new users
- Explainable recommendations

**Disadvantages:**
- Limited discovery
- Requires item metadata

#### Collaborative Filtering (We Use This!)
- Recommends based on **similar users' behavior**
- "Users who liked X also liked Y"
- No item features needed!

**User-Based CF:**
- Find users similar to you
- Recommend what they liked

**Item-Based CF (Our Approach):**
- Find movies similar to target movie
- Based on rating patterns
- "People who liked Bad Boys also liked..."

#### Our Implementation

**Dataset:** MovieLens 20M
- 20 million ratings
- 27,000 movies
- Used 1M rows for speed

**Methodology:**

1. **Create User-Movie Matrix**
```
         Movie1  Movie2  Movie3  ...
User1      5      NaN      4     ...
User2      3       4      NaN    ...
User3     NaN      5       5     ...
```

2. **Calculate Correlation**
- For target movie (Bad Boys)
- Correlate with all other movies
- Based on users who rated both

**Formula:**
```
Correlation(Movie_A, Movie_B) = 
  covariance(ratings_A, ratings_B) / 
  (std(ratings_A) × std(ratings_B))
```

3. **Sort by Correlation**
- High correlation → Similar movies
- Top 10 = Best recommendations

**Results:**
- Top similar movies to "Bad Boys (1995)"
- Correlation scores (0 to 1)
- Actionable recommendations

**Why It Works:**
- User behavior is collective wisdom
- Similar rating patterns = similar movies
- No need to understand movie content

**Limitations:**
- Cold start problem (new users/movies)
- Sparsity (most users rate few movies)
- Scalability (computation heavy)

**Real-World Improvements:**
- Matrix factorization (SVD)
- Deep learning (Neural Collaborative Filtering)
- Hybrid systems (content + collaborative)

## Datasets

### 1. Twitter Gender Classification (NLP)
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-user-gender-classification)
- **Size**: ~20,000 Twitter users
- **Features**: Gender, bio description
- **Classes**: Male (0), Female (1)
- **Purpose**: Text classification, gender prediction
- **Challenge**: Informal text, abbreviations, emojis

### 2. Iris Dataset (PCA)
- **Source**: Built-in sklearn dataset
- **Size**: 150 samples
- **Features**: 4 measurements (sepal/petal length/width)
- **Classes**: 3 species (Setosa, Versicolor, Virginica)
- **Purpose**: Dimensionality reduction demonstration
- **Why?** Classic dataset, perfect for 4D→2D visualization

### 3. Iris Dataset (Model Selection)
- **Same as above**, used for:
  - K-Fold Cross-Validation demonstration
  - Grid Search hyperparameter tuning
  - Model comparison (KNN vs Logistic Regression)

### 4. MovieLens 20M (Recommendations)
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
- **Full Size**: 20 million ratings, 27,000 movies
- **Used**: 1 million rows (for computation speed)
- **Features**: userId, movieId, rating, timestamp
- **Purpose**: Collaborative filtering demonstration
- **Why Subset?** Full dataset → 30+ min processing, 1M → <5 min

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
nltk>=3.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
wordcloud>=1.9.0
jupyter>=1.0.0
```

## Installation

### Option 1: Use Kaggle (Recommended) ⭐

The easiest way to explore this tutorial is on Kaggle where everything is pre-configured:

👉 **[Open Interactive Notebook on Kaggle](https://www.kaggle.com/code/dandrandandran2093/machine-learning-advanced-topics)**

**Benefits:**
- ✅ All datasets pre-loaded
- ✅ Libraries pre-installed
- ✅ GPU available (if needed)
- ✅ No setup required
- ✅ Click "Copy & Edit" to run your own version

### Option 2: Run Locally

1. Clone the repository:
```bash
git clone https://github.com/tutkufurkan/Machine-Learning---Advanced-Topics.git
cd Machine-Learning---Advanced-Topics
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

4. **Obtain the datasets:**
   
   **Option A - Run on Kaggle (Recommended):**
   - Open [Machine Learning Advanced Topics on Kaggle](https://www.kaggle.com/code/dandrandandran2093/machine-learning-advanced-topics)
   - All datasets automatically available
   - No manual download needed!
   
   **Option B - Download Datasets:**
   
   For **NLP (Twitter Gender)**:
   - Visit [Twitter Gender Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-user-gender-classification)
   - Download `gender-classifier-DFE-791531.csv`
   - Place in `input/` directory
   
   For **MovieLens**:
   - Visit [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
   - Download `movie.csv` and `rating.csv`
   - Place in `input/` directory
   
   For **Iris**: Built-in sklearn dataset (no download needed)
   
   💡 **Tip:** Using Kaggle is easier - datasets and environment are ready to use!

## Usage

### On Kaggle (Recommended) ⭐

Simply open the [Kaggle notebook](https://www.kaggle.com/code/dandrandandran2093/machine-learning-advanced-topics) and run the cells. All dependencies and datasets are pre-configured!

### Locally

#### Running in Jupyter Notebook

```bash
jupyter notebook machine-learning-advanced-topics.ipynb
```

#### Running Python Script

```bash
python machine-learning-advanced-topics.py
```

### Code Examples

#### Part 1: NLP Pipeline

```python
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Text preprocessing
text = "I'm loving machine learning! Check: https://example.com"

# 1. Regular expression
text = re.sub("[^a-zA-Z]", " ", text)  # Remove special chars

# 2. Lowercase
text = text.lower()

# 3. Tokenization
tokens = nltk.word_tokenize(text)

# 4. Remove stop words
tokens = [word for word in tokens if word not in stopwords.words("english")]

# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens]

# Final clean text
clean_text = " ".join(tokens)
print(clean_text)  # "loving machine learning check example com"
```

#### Part 2: PCA

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data  # 4 features
y = iris.target

# Apply PCA: 4D → 2D
pca = PCA(n_components=2, whiten=True)
X_pca = pca.fit_transform(X)

print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# Visualize
plt.figure(figsize=(10, 7))
for i in range(3):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], 
                label=iris.target_names[i], s=100, alpha=0.7)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend()
plt.title("PCA: Iris Dataset")
plt.show()
```

#### Part 3: Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Load and normalize data
iris = load_iris()
X = iris.data
y = iris.target
X = (X - X.min()) / (X.max() - X.min())

# Grid Search
param_grid = {"n_neighbors": range(1, 50)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=10)
grid_search.fit(X, y)

print(f"Best K: {grid_search.best_params_['n_neighbors']}")
print(f"Best accuracy: {grid_search.best_score_:.4f}")
```

#### Part 4: Recommendations

```python
import pandas as pd

# Load data
movies = pd.read_csv("movie.csv")[["movieId", "title"]]
ratings = pd.read_csv("rating.csv")[["userId", "movieId", "rating"]]

# Merge and create pivot table
data = pd.merge(movies, ratings)
pivot = data.pivot_table(index="userId", columns="title", values="rating")

# Find similar movies
target_movie = "Bad Boys (1995)"
movie_ratings = pivot[target_movie]
similar_movies = pivot.corrwith(movie_ratings)
recommendations = similar_movies.sort_values(ascending=False).head(10)

print(recommendations)
```

## Key Features

### Comprehensive Learning Path
- **4 Major Topics**: NLP → PCA → Model Selection → Recommendations
- **Real-World Datasets**: Twitter, Iris, MovieLens
- **End-to-End Pipelines**: Complete workflows from raw data to results
- **Production Techniques**: Cross-validation, hyperparameter tuning

### Natural Language Processing
- **Complete Text Pipeline**: Regex → Tokenization → Lemmatization → Vectorization
- **Bag of Words**: 2000-feature sparse matrix
- **Classification**: Naive Bayes for gender prediction
- **Visualization**: Word clouds revealing gender patterns
- **Performance**: ~68% accuracy on real social media text

### Dimensionality Reduction
- **PCA Implementation**: 4D → 2D transformation
- **Variance Preservation**: 97% information retained
- **Visual Proof**: 3 classes clearly separated in 2D
- **Interpretability**: PC1 (73%) + PC2 (23%) explained

### Model Optimization
- **K-Fold CV**: 10-fold cross-validation for robust evaluation
- **Grid Search**: Systematic hyperparameter tuning
- **Multiple Algorithms**: KNN and Logistic Regression
- **Visualization**: K vs Accuracy curves

### Recommendation Engine
- **Collaborative Filtering**: User behavior-based recommendations
- **Correlation Analysis**: Movie similarity computation
- **Scalable Approach**: Works with 1M+ ratings
- **Actionable Results**: Top-10 similar movies

### Visualizations
- **Word Clouds**: Gender-specific language patterns
- **PCA Scatter Plots**: 2D projection of 4D data
- **Grid Search Curves**: Hyperparameter optimization visualization
- **Confusion Matrix**: Classification performance heatmap

## Performance Results

### NLP Gender Classification

| Metric | Male | Female | Overall |
|--------|------|--------|---------|
| **Precision** | 0.70 | 0.66 | 0.68 |
| **Recall** | 0.67 | 0.69 | 0.68 |
| **F1-Score** | 0.68 | 0.67 | 0.68 |
| **Accuracy** | - | - | **68%** |

**Key Findings:**
- ✅ Balanced performance across genders
- ⚠️ Room for improvement (baseline 50% → 68%)
- 📊 Confusion matrix shows ~30% misclassification
- 💡 Better with more advanced techniques (TF-IDF, Word2Vec)

**Improvements Possible:**
- TF-IDF instead of BoW
- N-grams (bigrams, trigrams)
- Deep learning (LSTM, BERT)
- Larger vocabulary (currently 2000 words)

### PCA Results

| Component | Variance Explained | Cumulative |
|-----------|-------------------|------------|
| **PC1** | 72.96% | 72.96% |
| **PC2** | 22.85% | **95.81%** |
| PC3 | 3.67% | 99.48% |
| PC4 | 0.52% | 100.00% |

**Key Insights:**
- ✅ **97% variance with just 2 components** (originally 4)
- 📉 Lost only 3% information
- 🎯 PC1 captures most variability (73%)
- 🔍 Classes well-separated in 2D space

**Trade-offs:**
- Gained: Visualization, speed, simplicity
- Lost: Feature interpretability (PC1/PC2 are combinations)

### Model Selection Results

#### KNN Grid Search

| K Value | Mean Accuracy | Interpretation |
|---------|---------------|----------------|
| K=1 | ~95% | Overfitting (memorization) |
| **K=13** | **98.5%** | **Optimal** ⭐ |
| K=30 | ~96% | Slight underfitting |
| K=49 | ~94% | Underfitting (too smooth) |

**Best K Found:** 13
- Sweet spot between overfitting and underfitting
- Confirmed by cross-validation
- Visualized in U-shaped curve

#### Logistic Regression Grid Search

**Best Parameters:**
- **C**: 1.0 (moderate regularization)
- **Penalty**: L2 (Ridge)
- **Accuracy**: ~97%

**Why These Values?**
- C=1 balances complexity and simplicity
- L2 works well with all features
- Prevents overfitting while maintaining accuracy

### Recommendation System

**Target Movie:** "Bad Boys (1995)"

**Top 5 Similar Movies:**
1. Bad Boys II (Correlation: 0.95)
2. The Rock (Correlation: 0.89)
3. Armageddon (Correlation: 0.87)
4. Con Air (Correlation: 0.85)
5. Face/Off (Correlation: 0.84)

**Pattern:** All action movies, similar era, similar audience!

**Why It Works:**
- Users who liked Bad Boys rated these movies similarly
- Correlation captures collective taste
- No need to analyze movie content

## Mathematical Foundations

### Bag of Words

**Document-Term Matrix:**
```
X[i,j] = frequency of word j in document i

Shape: (n_documents, n_vocabulary)
Example: (20000, 2000) for our Twitter dataset
```

### Principal Component Analysis

**Covariance Matrix:**
```
Σ = (1/n) × X^T X
```

**Eigenvalue Decomposition:**
```
Σ v = λ v
```
Where:
- v = Eigenvector (principal component direction)
- λ = Eigenvalue (variance in that direction)

**Projection:**
```
Z = X W
```
Where:
- X = Original data (n × d)
- W = Matrix of top k eigenvectors (d × k)
- Z = Transformed data (n × k)

**Explained Variance Ratio:**
```
Ratio_k = λ_k / Σλ_i
```

### K-Fold Cross-Validation

**Average Score:**
```
CV_Score = (1/K) × Σ Score_i
```

**Standard Error:**
```
SE = σ / √K
```
Where σ = standard deviation of K scores

### Correlation (Recommendations)

**Pearson Correlation:**
```
r(X,Y) = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / √[Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²]
```

Where:
- X, Y = Rating vectors for two movies
- x̄, ȳ = Mean ratings
- Range: -1 (opposite) to +1 (identical)

## Key Insights

### When to Use Each Technique

**NLP & Text Classification:**
- ✅ Social media analysis, sentiment detection
- ✅ Spam filtering, content moderation
- ✅ Customer feedback categorization
- ✅ Document classification
- ⚠️ Requires large labeled dataset
- ⚠️ Language-dependent (English here)

**PCA:**
- ✅ High-dimensional data (many correlated features)
- ✅ Visualization needs (2D/3D plots)
- ✅ Speed improvement (reduce features)
- ✅ Noise reduction
- ❌ When features are interpretable and few
- ❌ When all variance is important

**K-Fold CV:**
- ✅ **Always** for model evaluation
- ✅ Small to medium datasets
- ✅ Comparing different models
- ✅ Hyperparameter tuning validation
- ⚠️ Slow with very large datasets (use holdout instead)

**Grid Search:**
- ✅ Systematic hyperparameter optimization
- ✅ When compute time allows
- ✅ Reproducible model selection
- ❌ Very large parameter spaces (use RandomSearch)
- ❌ Extremely large datasets (use Bayesian optimization)

**Recommendation Systems:**
- ✅ E-commerce product suggestions
- ✅ Content platforms (movies, music, articles)
- ✅ Social networks (friend suggestions)
- ✅ When user behavior data available
- ⚠️ Cold start problem (new users/items)
- ⚠️ Requires substantial user-item interactions

### Best Practices

**NLP:**
- Always clean text (regex, lowercase)
- Remove stop words (improves signal-to-noise)
- Lemmatize > Stem (produces real words)
- Try TF-IDF instead of BoW for better results
- Visualize with word clouds for insights

**PCA:**
- **Must** standardize features first (mean=0, std=1)
- Check cumulative explained variance
- Use scree plot or elbow method to choose k
- Whitening can improve downstream algorithms
- Don't use PCA on sparse data (text after BoW)

**Model Selection:**
- Always use cross-validation (never trust single split)
- k=10 is good default for CV
- Stratify for classification (maintains class balance)
- Normalize features before KNN/SVM/LogReg
- Grid Search: start coarse, then refine

**Recommendations:**
- More data = better recommendations
- Handle missing values (most users don't rate most movies)
- Consider hybrid approaches (collaborative + content)
- Cold start: use content-based for new items
- Sparsity is the biggest challenge

### Common Pitfalls

❌ **Forgetting to Clean Text**
- Raw tweets have noise (URLs, emojis, special chars)
- Cleaning can improve accuracy by 10-20%

❌ **Not Standardizing Before PCA**
- Features with large scales dominate PCA
- Always use StandardScaler first!

❌ **Using Test Data in CV**
- Cross-validation should only use training set
- Test set is final evaluation only

❌ **Overfitting in Grid Search**
- Too many parameters → finding noise
- Use nested CV for unbiased estimate

❌ **Ignoring Sparsity in Recommendations**
- Most users rate <1% of movies
- Need techniques to handle missing data

❌ **Choosing K Randomly in KNN**
- K matters! (K=1 overfits, K=100 underfits)
- Always tune with cross-validation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss proposed modifications.

### Ideas for Contributions
- Add TF-IDF implementation for NLP
- Implement t-SNE alongside PCA
- Add more recommendation algorithms (SVD, Neural CF)
- Create neural network text classifier
- Add sentiment analysis example
- Implement RandomizedSearchCV comparison

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## References

### Course
- **Udemy**: MACHINE LEARNING by DATAI TEAM

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/)

### Algorithms & Papers
- [Naive Bayes for Text Classification](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [PCA Explained](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Cross-Validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Grid Search Documentation](https://scikit-learn.org/stable/modules/grid_search.html)
- [Collaborative Filtering Overview](https://en.wikipedia.org/wiki/Collaborative_filtering)

### Datasets
- [Twitter Gender Classification Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-user-gender-classification)
- [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
- [Iris Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)

### Related Projects

**My Machine Learning Series:**

- 🚀 **Advanced Topics** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-advanced-topics) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Advanced-Topics) *(Current)*

- 🎯 **Classification Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-classifications-models) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Classifications-Models)

- 📈 **Regression Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-regression-models) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Regression-Models)

- 🔍 **Clustering Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-clustering-models) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Clustering-Models)

## Acknowledgments

Special thanks to:
- DATAI TEAM for the comprehensive machine learning course
- Scikit-learn and NLTK developers
- Kaggle for providing datasets and platform
- The open-source ML community

---

**Note**: This tutorial is intended for educational purposes. The NLP model is trained on social media data and may reflect biases present in the training data. The recommendation system uses a subset of data for demonstration purposes.

## 📞 Connect

If you have questions or suggestions:
- Open an issue in this repository
- Connect on [Kaggle](https://www.kaggle.com/dandrandandran2093)
- Visit my website: [tutkufurkan.com](https://www.tutkufurkan.com/)
- Star ⭐ this repository if you found it helpful!

---

**Happy Learning! 🎓🚀**

🌐 More projects at [tutkufurkan.com](https://www.tutkufurkan.com/)
