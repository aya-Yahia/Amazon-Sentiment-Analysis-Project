# ==========================================================
# STEP 1: DATA COLLECTION
# ==========================================================
# install.packages(c("readr", "quanteda", "quanteda.textmodels", "ggplot2", 
#                   "reshape2", "randomForest", "rpart", "rpart.plot", "pROC"))

library(readr)
library(quanteda)
library(quanteda.textmodels)
library(ggplot2)
library(reshape2)
library(randomForest)
library(rpart)
library(rpart.plot)
library(pROC)

raw_data <- read_csv("Amazon_Reviews.csv")
review_corpus <- corpus(raw_data, text_field = "Review Text")

# ==========================================================
# STEP 2: DATA PREPROCESSING
# ==========================================================
review_tokens <- tokens(review_corpus, 
                        remove_punct = TRUE, 
                        remove_symbols = TRUE, 
                        remove_numbers = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("english")) %>%
  tokens_wordstem()

# ==========================================================
# STEP 3: LABEL GENERATION (RULE-BASED)
# ==========================================================
# 1 -> Potentially Fake, 0 -> Genuine 
raw_data$target_label <- 0 

# Rule A: Short length + High Rating
word_counts <- ntoken(review_tokens)
raw_data$target_label[grepl("Rated 5 out of 5", raw_data$Rating) & word_counts < 10] <- 1

# Rule B: Duplicates
raw_data$target_label[duplicated(raw_data$`Review Text`)] <- 1

# Rule C: Contradictions
pos_words <- c("excel", "amaz", "perfect", "love", "best", "great")
pos_count <- ntoken(tokens_select(review_tokens, pattern = pos_words))
raw_data$target_label[grepl("Rated 1 out of 5", raw_data$Rating) & pos_count > 3] <- 1

# ==========================================================
# STEP 4: FEATURE EXTRACTION & DATA SPLITTING
# ==========================================================
# 4.1 Create the TF-IDF Matrix
review_dfm <- dfm(review_tokens) %>% dfm_trim(min_docfreq = 5)
review_tfidf <- dfm_tfidf(review_dfm)

# 4.2 Attach labels to metadata
docvars(review_corpus, "is_fake") <- raw_data$target_label

# 4.3 Split Data into Training (80%) and Testing (20%)
set.seed(123)
train_idx <- sample(1:ndoc(review_tfidf), 0.8 * ndoc(review_tfidf))

dfm_train <- review_tfidf[train_idx, ]
dfm_test <- review_tfidf[-train_idx, ]

labels <- docvars(review_corpus, "is_fake")
labels_train <- labels[train_idx]
labels_test <- labels[-train_idx]

#  MODEL TRAINING & COMPARISON
# ==========================================================

# --- 1. NAÏVE BAYES (Primary Model) ---
model_nb <- textmodel_nb(dfm_train, labels_train)
pred_nb <- predict(model_nb, newdata = dfm_test)

# --- 2. LOGISTIC REGRESSION ---
model_logit <- textmodel_lr(dfm_train, labels_train)
pred_logit <- predict(model_logit, newdata = dfm_test)

# --- 3. DECISION TREE (Optimized for Visualization) ---
clean_tree <- rpart(labels_train ~ ., 
                    data = as.data.frame(as.matrix(dfm_train[, 1:100])), 
                    method = "class",
                    control = rpart.control(maxdepth = 4, cp = 0.01))
pred_tree <- predict(clean_tree, newdata = as.data.frame(as.matrix(dfm_test[, 1:100])), type = "class")

# --- 4. RANDOM FOREST (Advanced Comparison) ---
# Prepare data with clean names to avoid "object not found" error
train_df_rf <- as.data.frame(as.matrix(dfm_train[, 1:100]))
colnames(train_df_rf) <- make.names(colnames(train_df_rf))
train_df_rf$label <- as.factor(labels_train)

test_df_rf <- as.data.frame(as.matrix(dfm_test[, 1:100]))
colnames(test_df_rf) <- make.names(colnames(test_df_rf))

model_rf <- randomForest(label ~ ., data = train_df_rf, ntree = 100)
pred_rf <- predict(model_rf, newdata = test_df_rf)

# ==========================================================
# STEP 7: EVALUATION & COMPARISON TABLE
# ==========================================================
acc_nb <- sum(pred_nb == labels_test) / length(labels_test)
acc_logit <- sum(pred_logit == labels_test) / length(labels_test)
acc_tree <- sum(pred_tree == labels_test) / length(labels_test)
acc_rf <- sum(pred_rf == labels_test) / length(labels_test)

comparison_results <- data.frame(
  Algorithm = c("Naïve Bayes", "Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = round(c(acc_nb, acc_logit, acc_tree, acc_rf), 4)
)
print(comparison_results)

# Detailed Metrics for Naïve Bayes (Primary)
results_table <- table(Predicted = pred_nb, Actual = labels_test)
precision <- results_table[2,2] / sum(results_table[2,])
recall <- results_table[2,2] / sum(results_table[,2])
f1_score <- 2 * ((precision * recall) / (precision + recall))

print(paste("F1-Score (Naïve Bayes):", round(f1_score, 4)))

# Prepare the data for plotting
# (Melting turns it from wide format to long format, required by ggplot)
melted_comparison <- melt(comparison_results, id.vars = "Algorithm")

# Define professional academic colors (Blues)
# or you can use your high-contrast burgundy: "#B21F49"
color_palette <- scale_fill_manual(values = c("#3498db", "#2980b9", "#1f618d", "#1a5276"))

ggplot(melted_comparison, aes(x = Algorithm, y = value, fill = Algorithm)) +
  geom_bar(stat = "identity", width = 0.7, color = "black") +
  
  # Add the accuracy percentage labels on top of the bars
  geom_text(aes(label = paste0(round(value * 100, 1), "%")), 
            vjust = -0.5, size = 5, fontface = "bold") +
  
  # Styling and theme
  theme_minimal() +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1.05)) +
  # Use manual color palette (optional) or standard ggplot colors
  # color_palette +
  
  labs(
    title = "Benchmark Analysis: Classifier Performance Comparison",
    subtitle = "Comparing accuracy scores of four distinct algorithms",
    x = "Machine Learning Algorithm",
    y = "Accuracy Score (0 to 1)"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "gray50"),
    axis.title = element_text(face = "bold", size = 12),
    axis.text.x = element_text(size = 11, angle = 15, vjust = 0.5), # Tilt labels for readability
    legend.position = "none" # Remove legend as X-axis labels are sufficient
  )

# ==========================================================
# STEP 8: VISUALIZATIONS FOR REPORT
# ==========================================================

# 1. Confusion Matrix Heatmap
melted_results <- as.data.frame(results_table)
ggplot(melted_results, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "#B21F49") +
  geom_text(aes(label = Freq), size = 5) +
  theme_minimal() +
  labs(title = "Confusion Matrix: Naïve Bayes Results")

# 2. Donut Chart: Class Distribution
dist_df <- as.data.frame(table(raw_data$target_label))
colnames(dist_df) <- c("Category", "Count")
ggplot(dist_df, aes(x = 2, y = Count, fill = Category)) +
  geom_bar(stat = "identity") +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  theme_void() +
  labs(title = "Distribution of Genuine (0) vs Fake (1)")

# 3. Decision Tree Logic Map
par(mar = c(1, 1, 1, 1)) 
rpart.plot(clean_tree, 
           type = 4, extra = 101, 
           box.palette = "RdYlGn", 
           main = "Decision Logic Tree")

# 4. ROC Curve
roc_nb <- roc(labels_test, as.numeric(pred_nb))
plot(roc_nb, col = "#3498db", lwd = 3, main = "ROC Curve: Performance Analysis")
abline(a = 0, b = 1, lty = 2, col = "grey")


