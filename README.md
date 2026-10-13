# Amazon-Sentiment-Analysis-Project
A deep dive into the 'unstructured' world of consumer language. Using Natural Language Processing (NLP) techniques in R, I transformed thousands of qualitative feedback entries into quantitative insights. By decoding sentiment trends and semantic patterns, I uncovered the hidden 'voice of the customer' to drive brand strategy

# 🛒 Amazon Reviews Sentiment Analysis Pipeline

## 📌 Project Overview
This repository presents a complete **Natural Language Processing (NLP) & Machine Learning Pipeline** constructed in **R** for sentiment classification of Amazon customer reviews. 

The system handles raw, unstructured review text and executes an automated end-to-end workflow: **Text Cleaning ➔ Unsupervised Lexicon Labeling ➔ Class Balancing ➔ Document-Term Matrix (DTM) Feature Engineering ➔ Benchmark ML Classification**.

---

## 🛠️ Tech Stack & Libraries
* **Language:** R
* **NLP & Text Mining:** `tidytext`, `tm`, `stringr`, `SnowballC`, `wordcloud`
* **Data Wrangling:** `tidyverse` (`dplyr`, `ggplot2`), `janitor`
* **Machine Learning & Benchmarking:** `caret`, `e1071` (Naïve Bayes), `ranger` (Random Forest), `rpart` (Decision Trees), `class` (KNN), `pROC`

---

## ⚙️ Methodology & Pipeline Stages

1. **Text Preprocessing & Quality Control:**
   * Consolidated `review_title` and `review_text` into a single body text.
   * Lowercased text, stripped URLs, removed non-alphabetic characters, and removed reviews <20 characters or exact duplicates.
2. **Ground-Truth Sentiment Labeling (Bing Lexicon):**
   * Tokenized clean text and calculated sentiment score: $\text{Score} = \text{Positive Words} - \text{Negative Words}$.
   * Applied binary classification: $\ge 0 \rightarrow \text{Positive}$, $< 0 \rightarrow \text{Negative}$.
3. **Data Balancing & Partitioning:**
   * Applied **undersampling** to achieve a balanced dataset (~19,400 reviews; 9,700 per class).
   * Split data into **80% Training** and **20% Testing** sets.
4. **Feature Engineering (DTM Construction):**
   * Built a Document-Term Matrix (DTM) from the clean training corpus after removing standard English stopwords.
   * Managed sparsity by stripping terms appearing in less than **0.5% of documents** to prevent overfitting.

---

## 📊 Exploratory Data Analysis & Visualizations

### 1. Class Distribution & Word Frequencies
![Class Distribution](screenshots/01_class_distribution.jfif)
![Top Words](screenshots/02_top_words.png)

### 2. Side-by-Side Word Clouds
![Word Clouds](screenshots/03_wordclouds.png)

### 3. Bigram Contextual Analysis
* **Positive Bigrams:** Dominated by phrases like `love amazon`, `customer support`, and `fast delivery`.
* **Negative Bigrams:** Dominated by service friction drivers like `customer service`, `amazon prime`, `gift card`, and `prime membership`.

| Positive Bigrams | Negative Bigrams |
|---|---|
| ![Positive Bigrams](screenshots/04_positive_bigrams.png) | ![Negative Bigrams](screenshots/05_negative_bigrams.png) |

---

## 🏆 Machine Learning Benchmarking & Performance

Five supervised classifiers were evaluated on the same 80/20 train/test split:

| Rank | Model | Accuracy | F1-Score | AUC | Key Strengths & Notes |
|:---:|---|:---:|:---:|:---:|---|
| 🥇 | **Logistic Regression** | **86.4%** | **0.88** | **~0.95** | **Best Overall Model:** Excellent linear separator on sparse DTM features. |
| 🥈 | **Random Forest (300 trees)** | **79.8%** | **0.78** | **0.89** | High stability and strong resistance to overfitting. |
| 🥉 | **Naïve Bayes** | **70.9%** | **0.75** | **0.75** | Fast, reliable probabilistic baseline. |
| 4 | **Decision Tree (`rpart`)** | **66.8%** | **0.68** | **0.72** | Highly interpretable, but prone to variance on high-dim text. |
| 5 | **KNN ($k=7$)** | **62.9%** | **0.66** | **0.50** | Degrades in high dimensions due to distance calculation limitations. |

### Performance Visual Comparison

| ROC Curves Comparison | Confusion Matrices Comparison |
|---|---|
| ![ROC Curves](screenshots/06_roc_curves.webp) | ![Confusion Matrices](screenshots/07_confusion_matrices.webp) |

---

## 💡 Key Strategic & Business Insights
* **Service Over Product Focus:** Frequent appearance of `delivery`, `service`, and `customer` across both sentiment classes reveals that customer dissatisfaction is primarily driven by **fulfillment logistics and support interaction**, rather than product quality alone.
* **Operational Interventions:** High occurrence of `late delivery` or `refund process` in negative bigrams provides actionable intelligence for streamlining return policies and logistics SLAs.

---

## 👥 Authors
* **Aya YAHIA** & **Israa HAMMAMI**
* **Institution:** Tunis Business School (Data Mining / Text Mining Course)
* **Date:** April 2026





