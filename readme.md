# Sentiment Analysis Web App & Notebook

## Table of Contents

* [Description](#description)
* [Progress](#progress)
* [Model Performance](#model-performance)
* [Installation](#installation)
* [Usage](#usage)
* [Libraries Used](#libraries-used)
* [Conclusion](#conclusion)

---

## Description

Social media platforms are a powerful source for understanding public opinion and brand perception. This project explores the use of **Artificial Intelligence (AI)** for sentiment analysis to automatically classify social media posts as **positive, negative, or neutral**. The workflow includes data collection, preprocessing, modeling, evaluation, and deployment in an interactive web application.

---

## Our Progress

The project was divided into several stages:

### 1. Data Exploration

* Loaded data into a pandas DataFrame (732 rows, 14 columns).
* Explored each column to identify features relevant for analysis.
* Detected 20 redundant columns and inconsistencies caused by extra spaces in string data.

### 2. Data Cleaning

* Removed redundant columns to reduce noise.
* Extracted multiple hashtags from single columns into separate fields.

### 3. Exploratory Data Analysis (EDA)

* Applied visualizations including pie charts, word clouds, line plots, scatter plots, and bar charts.
* Analyzed user activity and trends across Facebook, Instagram, and Twitter.

### 4. Text Preprocessing

* Implemented `clean_text` function using regex and NLTK for tokenization, stemming, stopword removal, and punctuation cleaning.
* Vectorized text using **Bag of Words** and **TF-IDF**.

### 5. Modeling

* Trained models: Logistic Regression, K-Nearest Neighbors, SVM, Naive Bayes, Decision Tree, Random Forest.
* Evaluated using **accuracy, precision, recall, and F1-score**.
* Trained models on both Bag of Words and TF-IDF features.

### 6. Deployment

* Selected SVM and Naive Bayes as the most reliable models.
* Provided an interactive web app allowing users to select models, input text and hashtags, and get real-time predictions with confidence scores.

---

## Model Performance

### Bag of Words (BOW)

| Model                   | Train Accuracy (%) | Test Accuracy (%) |
| ----------------------- | ------------------ | ----------------- |
| Logistic Regression     | 100.00             | 85.31             |
| SVM                     | 100.00             | 83.92             |
| Multinomial Naive Bayes | 97.36              | 82.52             |

### TF-IDF

| Model                   | Train Accuracy (%) | Test Accuracy (%) |
| ----------------------- | ------------------ | ----------------- |
| Multinomial Naive Bayes | 98.24              | 85.31             |
| SVM                     | 98.24              | 84.62             |
| KNN                     | 90.49              | 81.82             |

---

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the environment:

```bash
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Notebook

* Open `sentiment_analysis.ipynb`.
* Perform data exploration, preprocessing, model training, and evaluation.

### Streamlit App

```bash
streamlit run app.py
```

* Select a **feature extraction type** (BOW or TF-IDF) and model.
* Enter text and optional hashtags.
* Click **Analyze Sentiment** to view predictions and confidence.

---

## Libraries Used

* pandas, numpy
* matplotlib, seaborn, plotly, wordcloud
* scikit-learn, joblib
* nbformat

---

## Conclusion

* **Bag of Words (BOW):** Logistic Regression and SVM achieved top test accuracies (85.31% and 83.92%). Multinomial Naive Bayes performed well (82.52%), while tree-based models overfit despite perfect training scores.
* **TF-IDF:** Multinomial Naive Bayes achieved highest test accuracy (85.31%), slightly better than SVM (84.62%). KNN and Logistic Regression had moderate performance, while Decision Tree and Random Forest overfit.

### Overall Insights

* Multinomial Naive Bayes and SVM are the most reliable models for this dataset.
* Both Bag of Words and TF-IDF produced similar top accuracies, with TF-IDF slightly improving Naive Bayes performance.
* Tree-based models overfit, showing high training accuracy but poor generalization.
* KNN performed moderately but lagged behind top models.

This project demonstrates a **full workflow** from data exploration to deployment with **interactive visualization and real-time sentiment prediction**. Recommended models for deployment:

* Logistic Regression (BOW) ⭐
* Multinomial Naive Bayes (TF-IDF) ⚡
* SVM (TF-IDF) 🧠
