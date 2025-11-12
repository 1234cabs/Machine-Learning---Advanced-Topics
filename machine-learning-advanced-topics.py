#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-11T08:14:16.556Z
"""

# # 🎓 Machine Learning - Advanced Topics
# 
# ## 📋 INSTRUCTION
# 
# ### 📝 Part 1: Natural Language Processing (NLP)
# 1. [Introduction to NLP](#1)
# 2. [Regular Expression (RE)](#2)
# 3. [Stop Words Removal](#3)
# 4. [Lemmatization](#4)
# 5. [Data Cleaning Pipeline](#5)
# 6. [Bag of Words](#6)
# 7. [Text Classification](#7)
# 
# ### 📊 Part 2: Principal Component Analysis (PCA)
# 8. [Introduction to PCA](#8)
# 9. [PCA Implementation](#9)
# 10. [2D Visualization](#10)
# 
# ### 🎯 Part 3: Model Selection
# 11. [K-Fold Cross Validation](#11)
# 12. [Grid Search with KNN](#12)
# 13. [Grid Search with Logistic Regression](#13)
# 
# ### 🎬 Part 4: Recommendation Systems
# 14. [Introduction to Recommendation Systems](#14)
# 15. [Collaborative Filtering](#15)


# ## Import Libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud

# ML
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
print('✅ Libraries loaded!')

# ---
# 
# ## 📁 Datasets Used
# 
# **Part 1 - NLP:**
# - Twitter Gender Classification (~20K users, 2 classes)
# - [Dataset Link](https://www.kaggle.com/datasets/crowdflower/twitter-user-gender-classification)
# 
# **Part 2 - PCA:**
# - Iris Dataset (150 samples, 4 features, 3 classes)
# - Built-in sklearn dataset
# 
# **Part 4 - Recommendations:**
# - MovieLens 20M (20M ratings, 27K movies)
# - Used 1M rows for computation speed
# - [Dataset Link](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
# 
# ---


# ---
# # 📝 Part 1: Natural Language Processing (NLP)
# 
# **What we'll learn:** Text preprocessing pipeline, Bag of Words, Naive Bayes classification
# 
# **Dataset:** Twitter gender classification (~20K users)
# 
# **Goal:** Predict gender from bio text
# 
# ---


# <a id="1"></a>
# ## Introduction to NLP
# 
# **Natural Language Processing (NLP)** helps computers understand human language.
# 
# **Applications:** Spam detection, sentiment analysis, chatbots, translation
# 
# **This Project:** Gender classification from Twitter bios
# 
# **Pipeline:**
# ```
# Raw Text → Regex → Tokenization → Lemmatization → Bag of Words → Classification
# ```


# Load dataset
data = pd.read_csv(
    r"/kaggle/input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",
    encoding="latin1"
)

data.head()

data.info()

# Prepare data
data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis=0, inplace=True)
data.reset_index(drop=True, inplace=True)

# Encode: 0=male, 1=female
data.gender = [1 if each == "female" else 0 for each in data.gender]

data.gender.value_counts()

# <a id="2"></a>
# ## Regular Expression (RE)
# 
# **Purpose:** Remove special characters, numbers, URLs
# 
# **Pattern:** `[^a-zA-Z]` → Keep only letters


# Demo preprocessing
first_description = data.description.iloc[4]
print(f"Original: {first_description}\n")

description = re.sub("[^a-zA-Z]", " ", first_description)
print(f"After regex: {description}\n")

description = description.lower()
print(f"Lowercase: {description}")

# <a id="3"></a>
# ## Stop Words Removal
# 
# **Stop Words:** Common words with little meaning ("the", "is", "and")
# 
# **Why?** Reduce noise, focus on meaningful words


# Tokenization
description = nltk.word_tokenize(description)
print(f"Tokens: {description}\n")

# Remove stop words
description = [word for word in description if word not in set(stopwords.words("english"))]
print(f"After removal: {description}")

# <a id="4"></a>
# ## Lemmatization
# 
# **Convert words to root form:** "running" → "run", "better" → "good"


# Lemmatization
lemmatizer = WordNetLemmatizer()
description = [lemmatizer.lemmatize(word) for word in description]
description = " ".join(description)

print(f"Final: {description}")

# <a id="5"></a>
# ## Data Cleaning Pipeline


# Apply to all data
lemmatizer = WordNetLemmatizer()
description_list = []

for description in data.description:
    description = re.sub("[^a-zA-Z]", " ", description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [lemmatizer.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)

print(f"✅ Processed {len(description_list)} descriptions")

# <a id="6"></a>
# ## Bag of Words
# 
# **Converts text → numerical vectors**
# 
# Creates sparse matrix: documents × words


# Create BoW matrix
max_features = 2000
count_vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
sparse_matrix = count_vectorizer.fit_transform(description_list)

print(f"Shape: {sparse_matrix.shape}")
print(f"Sparsity: {(1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])) * 100:.2f}%")

# <a id="7"></a>
# ## Text Classification
# 
# **Algorithm:** Naive Bayes (fast, effective for text)


# Train-test split
y = data.iloc[:, 0].values
x = sparse_matrix

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {x_train.shape[0]}, Test: {x_test.shape[0]}")

# Train Naive Bayes
nb = MultinomialNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Male', 'Female'],
            yticklabels=['Male', 'Female'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(classification_report(y_test, y_pred, target_names=['Male', 'Female']))

# Word Clouds
male_text = " ".join([description_list[i] for i in range(len(data)) if data.gender.iloc[i] == 0])
female_text = " ".join([description_list[i] for i in range(len(data)) if data.gender.iloc[i] == 1])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

wc_male = WordCloud(width=800, height=400, background_color='white', 
                     colormap='Blues', max_words=100).generate(male_text)
axes[0].imshow(wc_male)
axes[0].set_title('Male Bios', fontsize=14, fontweight='bold')
axes[0].axis('off')

wc_female = WordCloud(width=800, height=400, background_color='white', 
                       colormap='Reds', max_words=100).generate(female_text)
axes[1].imshow(wc_female)
axes[1].set_title('Female Bios', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# ### 🎯 NLP Key Takeaways
# 
# - Achieved ~68% accuracy with Naive Bayes
# - Text preprocessing is crucial for performance
# - Word clouds reveal gender-specific patterns
# - Preprocessing quality > model complexity
# 
# ### 📚 Further Reading
# 
# - [NLTK Documentation](https://www.nltk.org/)
# - [Text Preprocessing Guide](https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing)
# - [Naive Bayes for Text](https://scikit-learn.org/stable/modules/naive_bayes.html)


# ---
# # 📊 Part 2: Principal Component Analysis (PCA)
# 
# **What we'll learn:** Dimensionality reduction while preserving variance
# 
# **Dataset:** Iris (150 samples, 4 features → 2 features)
# 
# **Goal:** Visualize 4D data in 2D, preserve 97% variance
# 
# ---


# <a id="8"></a>
# ## Introduction to PCA
# 
# **PCA reduces dimensions while preserving variance**
# 
# **Why?** Visualization, faster training, less overfitting
# 
# **How?** Find principal components (directions of max variance)


# <a id="9"></a>
# ## PCA Implementation


# Load Iris
iris = load_iris()
x_i = iris.data  # 4 features
y_i = iris.target

df = pd.DataFrame(x_i, columns=iris.feature_names)
df["class"] = y_i
df.head()

# PCA: 4D → 2D
pca = PCA(n_components=2, whiten=True)
pca.fit(x_i)
x_pca = pca.transform(x_i)

print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")
print(f"Lost: {(1-sum(pca.explained_variance_ratio_))*100:.1f}%")

# <a id="10"></a>
# ## 2D Visualization


# Add PCA to dataframe
df["PC1"] = x_pca[:, 0]
df["PC2"] = x_pca[:, 1]

# Visualize
plt.figure(figsize=(10, 7))
colors = ['red', 'green', 'blue']

for i in range(3):
    plt.scatter(df.PC1[df['class'] == i], df.PC2[df['class'] == i],
                color=colors[i], label=iris.target_names[i], s=100, alpha=0.7)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA: 4D → 2D")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ### 🎯 PCA Key Takeaways
# 
# - Reduced 4D → 2D, kept 97% variance
# - Classes well-separated in 2D space
# - PC1 explains ~73%, PC2 ~23%
# - Trade-off: simplicity vs interpretability
# 
# ### 📚 Further Reading
# 
# - [PCA Explained Visually](http://setosa.io/ev/principal-component-analysis/)
# - [Scikit-learn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
# - [When to Use PCA](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)


# ---
# # 🎯 Part 3: Model Selection
# 
# **What we'll learn:** K-Fold CV, Grid Search for hyperparameter tuning
# 
# **Dataset:** Iris (for demonstration)
# 
# **Goal:** Find optimal hyperparameters (K for KNN, C for LogReg)
# 
# ---


# <a id="11"></a>
# ## K-Fold Cross Validation
# 
# **Problem:** Single split → unreliable results
# 
# **Solution:** K-Fold (usually K=10)
# - Split data into K folds
# - Train on K-1, test on 1
# - Repeat K times, average results


# Prepare data
iris_k = load_iris()
x_k = iris_k.data
y_k = iris_k.target

# Normalize
x_k = (x_k - x_k.min()) / (x_k.max() - x_k.min())

x_k_train, x_k_test, y_k_train, y_k_test = train_test_split(
    x_k, y_k, test_size=0.3, random_state=42, stratify=y_k
)

# 10-Fold CV
knn = KNeighborsClassifier(n_neighbors=3)
accuracies = cross_val_score(estimator=knn, X=x_k_train, y=y_k_train, cv=10)

print(f"Mean accuracy: {np.mean(accuracies):.4f}")
print(f"Std: {np.std(accuracies):.4f}")

knn.fit(x_k_train, y_k_train)
print(f"Test accuracy: {knn.score(x_k_test, y_k_test):.4f}")

# <a id="12"></a>
# ## Grid Search with KNN
# 
# **Find optimal K value** using cross-validation


# Grid Search
grid = {"n_neighbors": np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=10)
knn_cv.fit(x_k, y_k)

print(f"Best K: {knn_cv.best_params_['n_neighbors']}")
print(f"Best accuracy: {knn_cv.best_score_:.4f}")

# Visualize
results = pd.DataFrame(knn_cv.cv_results_)

plt.figure(figsize=(12, 6))
plt.plot(results['param_n_neighbors'], results['mean_test_score'], marker='o')
plt.axvline(x=knn_cv.best_params_['n_neighbors'], color='red', 
            linestyle='--', label=f"Best K = {knn_cv.best_params_['n_neighbors']}")
plt.xlabel('K (Neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN: K vs Accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# <a id="13"></a>
# ## Grid Search with Logistic Regression
# 
# **Find optimal C (regularization) and penalty**


# Prepare subset
x_lr = x_k[:100, :]
y_lr = y_k[:100]

x_lr = (x_lr - x_lr.min()) / (x_lr.max() - x_lr.min())
x_lr_train, x_lr_test, y_lr_train, y_lr_test = train_test_split(
    x_lr, y_lr, test_size=0.3, random_state=42, stratify=y_lr
)

# Grid Search
grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
logreg = LogisticRegression(max_iter=1000, solver='saga')
logreg_cv = GridSearchCV(logreg, grid, cv=10)
logreg_cv.fit(x_lr, y_lr)

print(f"Best C: {logreg_cv.best_params_['C']}")
print(f"Best penalty: {logreg_cv.best_params_['penalty']}")
print(f"Best score: {logreg_cv.best_score_:.4f}")

# Test
logreg_best = LogisticRegression(C=logreg_cv.best_params_['C'], 
                                  penalty=logreg_cv.best_params_['penalty'],
                                  solver='saga', max_iter=1000)
logreg_best.fit(x_lr_train, y_lr_train)
print(f"Test accuracy: {logreg_best.score(x_lr_test, y_lr_test):.4f}")

# ### 🎯 Model Selection Key Takeaways
# 
# - K-Fold CV provides robust performance estimates
# - Grid Search finds optimal hyperparameters systematically
# - Found best K for KNN and best C/penalty for LogReg
# - Always use cross-validation for model selection
# 
# ### 📚 Further Reading
# 
# - [Cross-Validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
# - [Hyperparameter Tuning](https://www.kaggle.com/code/willkoehrsen/intro-to-model-tuning-grid-and-random-search)
# - [Grid Search vs Random Search](https://www.blog.trainindata.com/grid-search-vs-random-search-which-one-should-you-use/)


# ---
# # 🎬 Part 4: Recommendation Systems
# 
# **What we'll learn:** Collaborative filtering using correlation
# 
# **Dataset:** MovieLens 20M (used 1M rows)
# 
# **Goal:** Recommend similar movies based on user ratings
# 
# ---


# <a id="14"></a>
# ## Introduction to Recommendation Systems
# 
# **Suggest items based on preferences**
# 
# **Types:**
# - Content-Based: Similar items
# - Collaborative Filtering: Similar users (we'll use this)
# 
# **Applications:** Netflix, Amazon, Spotify


# <a id="15"></a>
# ## Collaborative Filtering


# Load MovieLens
movie = pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv")
movie = movie[["movieId", "title"]]

rating = pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv")
rating = rating[["userId", "movieId", "rating"]]

movie.head()

rating.head()

# Merge and subset (1M rows for speed)
data = pd.merge(movie, rating)
data = data.iloc[:1000000, :]

data.head(10)

# Create user-movie matrix
pivot_table = data.pivot_table(index=["userId"], columns=["title"], values="rating")

print(f"Shape: {pivot_table.shape}")
pivot_table.head(10)

# Find similar movies to "Bad Boys (1995)"
movie_watched = pivot_table["Bad Boys (1995)"]
similarity = pivot_table.corrwith(movie_watched)
similarity = similarity.sort_values(ascending=False)

similarity.head(10)

# ### 🎯 Recommendation Systems Key Takeaways
# 
# - Collaborative filtering uses user behavior patterns
# - Correlation measures similarity between items
# - Successfully recommended similar movies
# - Used 1M rows from 20M total for speed
# - Real-world systems use hybrid approaches
# 
# ### 📚 Further Reading
# 
# - [Recommendation Systems Guide](https://www.kaggle.com/code/kanncaa1/recommendation-systems-tutorial)
# - [Collaborative Filtering Explained](https://www.ibm.com/think/topics/collaborative-filtering?)
# - [Matrix Factorization Techniques](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)


# ---
# # 🎓 Summary
# ---
# 
# ## What We Learned
# 
# **NLP:** Text preprocessing → BoW → Classification
# 
# **PCA:** 4D → 2D, kept 97% variance
# 
# **Model Selection:** K-Fold CV + Grid Search for optimal hyperparameters
# 
# **Recommendations:** Collaborative filtering with correlation
# 
# ---


# # 🔗 References
# 
# ## 📚 My Machine Learning Series
# 
# This notebook is part of a comprehensive Machine Learning series:
# 
# | Notebook | Topics Covered |
# |----------|----------------|
# | 🔬 **Advanced Topics** | NLP, PCA, Model Selection, Recommendations *(Current)* |
# | 🔍 **Clustering Models** | [Link](https://www.kaggle.com/code/dandrandandran2093/machine-learning-clustering-models) - K-Means, Hierarchical Clustering |
# | 🎯 **Classification Models** | [Link](https://www.kaggle.com/code/dandrandandran2093/machine-learning-classifications-models) - Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest |
# | 📈 **Regression Models** | [Link](https://www.kaggle.com/code/dandrandandran2093/machine-learning-regression-models) - Linear, Polynomial, Decision Tree, Random Forest |
# 
# ---
# 
# **Course:** Udemy - MACHINE LEARNING by DATAI TEAM
# 
# **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, NLTK, WordCloud