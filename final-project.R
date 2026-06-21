install.packages("rpart")
install.packages("rpart.plot")
install.packages("randomForest")
install.packages("ROCR")

library(rpart)
library(rpart.plot)
library(randomForest)
library(ROCR)

tic.train <- read.csv("C:/Users/misal/OneDrive/Desktop/DAT-640/TicdataTraining.csv")
tic.test <-  read.csv("C:/Users/misal/OneDrive/Desktop/DAT-640/TicdataTesting.csv")


str(tic.train)
summary(tic.train)

names(tic.train)
plot(tic.train[, 1:10])

# Individual Correlations
correlations <- cor(tic.train, use = "complete.obs")[, "CARAVAN"]
correlations <- sort(correlations, decreasing = TRUE)
correlations <- correlations[names(correlations) != "CARAVAN"]
head(correlations, 10)
tail(correlations, 10)

# Individual Correlations Barchart
top_corr <- sort(abs(correlations), decreasing = TRUE)[1:10]
par(mar = c(8,4,4,2))
barplot(
  top_corr,
  las = 2,
  main = "Top Correlations with CARAVAN",
  ylab = "Absolute Correlation"
)

# Full Descriptive Stats + Statistical references
descriptive_stats <- data.frame(
  Variable = names(tic.train),
  Mean = sapply(tic.train, mean, na.rm = TRUE),
  Median = sapply(tic.train, median, na.rm = TRUE),
  SD = sapply(tic.train, sd, na.rm = TRUE),
  Variance = sapply(tic.train, var, na.rm = TRUE),
  Min = sapply(tic.train, min, na.rm = TRUE),
  Max = sapply(tic.train, max, na.rm = TRUE),
  Skewness = sapply(tic.train, function(x) {
    x <- na.omit(x)
    m <- mean(x)
    s <- sd(x)
    mean((x - m)^3) / s^3
  }),
  Kurtosis = sapply(tic.train, function(x) {
    x <- na.omit(x)
    m <- mean(x)
    s <- sd(x)
    mean((x - m)^4) / s^4
  })
)

# Top 10 by Standard Deviation
top10_sd <- descriptive_stats[order(descriptive_stats$SD, decreasing = TRUE),]
top10_sd <- top10_sd[top10_sd$Variable != "MOSTYPE",]
top10_sd <- head(top10_sd, 10)
top10_sd

# Top 10 Standard Deviation Barchart
par(mar = c(8,4,4,2))
barplot(
  top10_sd$SD,
  names.arg = top10_sd$Variable,
  las = 2,
  main = "Top 10 Variables by Standard Deviation",
  ylab = "Standard Deviation"
)

#===========================#
# Random Forest Development #
#===========================#
tic.model <- tic.train

tic.model$CARAVAN <- factor(
  tic.model$CARAVAN,
  levels = c(0, 1),
  labels = c("No", "Yes")
)

categorical_variables <- intersect(c("MOSTYPE", "MOSHOOFD"),names(tic.model))
for (variable in categorical_variables) {
  tic.model[[variable]] <- factor(tic.model[[variable]])
}

predictor_names <- setdiff(
  names(tic.model),
  "CARAVAN"
)

# Partition Function
create_stratified_partitions <- function(
    outcome,
    development_proportion = 0.60,
    validation_proportion = 0.20,
    seed = 08061996) {
  
  set.seed(seed)
  
  development_indices <- integer(0)
  validation_indices <- integer(0)
  
  for (outcome_level in levels(outcome)) {
    
    level_indices <- which(outcome == outcome_level)
    level_indices <- sample(level_indices)
    
    development_count <- floor(
      length(level_indices) * development_proportion
    )
    
    validation_count <- floor(
      length(level_indices) * validation_proportion
    )
    
    development_indices <- c(
      development_indices,
      level_indices[
        seq_len(development_count)
      ]
    )
    
    validation_start <- development_count + 1
    validation_end <- development_count + validation_count
    
    validation_indices <- c(
      validation_indices,
      level_indices[
        validation_start:validation_end
      ]
    )
  }
  
  internal_test_indices <- setdiff(
    seq_along(outcome),
    c(
      development_indices,
      validation_indices
    )
  )
  
  list(
    development = development_indices,
    validation = validation_indices,
    internal_test = internal_test_indices
  )
}

partitions <- create_stratified_partitions(tic.model$CARAVAN)

development.data <- tic.model[partitions$development,]
validation.data <- tic.model[partitions$validation,]
internal.test.data <- tic.model[partitions$internal_test,]

partition_distribution <- rbind(
  Development = prop.table(table(development.data$CARAVAN)),
  Validation = prop.table(table(validation.data$CARAVAN)),
  InternalTest = prop.table(table(internal.test.data$CARAVAN))
)

partition_distribution

# Model Evaluation Functions

safe_divide <- function(numerator, denominator) {
  
  if (
    length(denominator) == 0 ||
    is.na(denominator) ||
    denominator == 0
  ) {
    return(NA_real_)
  }
  
  numerator / denominator
}


calculate_auc <- function(actual, probability) {
  
  actual_numeric <- as.integer(actual == "Yes")
  
  prediction_object <- ROCR::prediction(
    probability,
    actual_numeric
  )
  
  auc_object <- ROCR::performance(
    prediction_object,
    measure = "auc"
  )
  
  as.numeric(
    auc_object@y.values[[1]]
  )
}


evaluate_model <- function(
    actual,
    probability,
    threshold,
    top_percentage = 0.10) {
  
  actual <- factor(
    actual,
    levels = c("No", "Yes")
  )
  
  predicted_class <- factor(
    ifelse(
      probability >= threshold,
      "Yes",
      "No"
    ),
    levels = c("No", "Yes")
  )
  
  confusion_matrix <- table(
    Actual = actual,
    Predicted = predicted_class
  )
  
  true_negative <- confusion_matrix["No", "No"]
  false_positive <- confusion_matrix["No", "Yes"]
  false_negative <- confusion_matrix["Yes", "No"]
  true_positive <- confusion_matrix["Yes", "Yes"]
  
  accuracy <- safe_divide(
    true_positive + true_negative,
    sum(confusion_matrix)
  )
  
  precision <- safe_divide(
    true_positive,
    true_positive + false_positive
  )
  
  recall <- safe_divide(
    true_positive,
    true_positive + false_negative
  )
  
  specificity <- safe_divide(
    true_negative,
    true_negative + false_positive
  )
  
  f1_score <- safe_divide(
    2 * precision * recall,
    precision + recall
  )
  
  auc <- calculate_auc(
    actual,
    probability
  )
  
  number_selected <- ceiling(
    length(probability) * top_percentage
  )
  
  ranked_indices <- order(
    probability,
    decreasing = TRUE
  )
  
  selected_indices <- ranked_indices[
    seq_len(number_selected)
  ]
  
  overall_response_rate <- mean(
    actual == "Yes"
  )
  
  selected_response_rate <- mean(
    actual[selected_indices] == "Yes"
  )
  
  lift <- safe_divide(
    selected_response_rate,
    overall_response_rate
  )
  
  gain <- safe_divide(
    sum(actual[selected_indices] == "Yes"),
    sum(actual == "Yes")
  )
  
  
  list(
    metrics = data.frame(
      Threshold = threshold,
      Accuracy = accuracy,
      Precision = precision,
      Recall = recall,
      Specificity = specificity,
      F1 = f1_score,
      AUC = auc,
      LiftTop10Percent = lift,
      GainTop10Percent = gain
    ),
    
    confusion_matrix = confusion_matrix
  )
}

minority_count <- as.integer(
  table(development.data$CARAVAN)["Yes"]
)

mtry_candidates <- c(9, 14, 16, 18)
majority_ratios <- c(2, 3, 4)
nodesize_candidates <- c(5, 10, 15, 20)

random_forest_candidates <- list()
random_forest_tuning <- data.frame()

for (current_mtry in mtry_candidates) {
  for (current_ratio in majority_ratios) {
    for (current_nodesize in nodesize_candidates) {
    
      majority_sample <- min(
        as.integer(table(development.data$CARAVAN)["No"]),
        floor(minority_count * current_ratio)
      )
      
      model_key <- paste(
        current_mtry,
        current_ratio,
        current_nodesize,
        sep = "_"
      )
      
      cat(
        "Training: mtry =", current_mtry,
        "ratio =", current_ratio,
        "nodesize =", current_nodesize,
        "\n"
      )
      
      set.seed(08061996)
      
      current_model <- randomForest(
        CARAVAN ~ .,
        data = development.data,
        ntree = 1000,
        mtry = current_mtry,
        nodesize = current_nodesize,
        importance = TRUE,
        strata = development.data$CARAVAN,
        sampsize = c(
          No = majority_sample,
          Yes = minority_count
        )
      )
      
      validation_probability <- predict(
        current_model,
        newdata = validation.data,
        type = "prob"
      )[, "Yes"]
      
      validation_auc <- calculate_auc(
        actual = validation.data$CARAVAN,
        probability = validation_probability
      )
      
      random_forest_candidates[[model_key]] <- current_model
      
      random_forest_tuning <- rbind(
        random_forest_tuning,
        data.frame(
          Key = model_key,
          mtry = current_mtry,
          MajorityRatio = current_ratio,
          NodeSize = current_nodesize,
          ValidationAUC = validation_auc
        )
      )
    }
  }
}

random_forest_tuning[
  order(random_forest_tuning$ValidationAUC, decreasing = TRUE),
]

select_balanced_threshold <- function(actual, probability) {
  
  thresholds <- seq(0.01, 0.99, by = 0.01)
  
  results <- data.frame(
    Threshold = thresholds,
    Recall = NA_real_,
    Specificity = NA_real_,
    BalancedAccuracy = NA_real_,
    F1 = NA_real_
  )
  
  for (i in seq_along(thresholds)) {
    
    current_results <- evaluate_model(
      actual = actual,
      probability = probability,
      threshold = thresholds[i]
    )
    
    results$Recall[i] <-
      current_results$metrics$Recall
    
    results$Specificity[i] <-
      current_results$metrics$Specificity
    
    results$BalancedAccuracy[i] <- mean(
      c(
        current_results$metrics$Recall,
        current_results$metrics$Specificity
      ),
      na.rm = TRUE
    )
    
    results$F1[i] <-
      current_results$metrics$F1
  }
  
  results[
    which.max(results$BalancedAccuracy),
  ]
}

best_row <- random_forest_tuning[
  which.max(random_forest_tuning$ValidationAUC),
]

best_row

random_forest_model <-
  random_forest_candidates[[best_row$Key]]

random_forest_validation_probability <- predict(
  random_forest_model,
  newdata = validation.data,
  type = "prob"
)[, "Yes"]

threshold_results <- select_balanced_threshold(
  actual = validation.data$CARAVAN,
  probability = random_forest_validation_probability
)

threshold_results
random_forest_threshold <- threshold_results$Threshold

#===========================#
# Random Forest Development #
#===========================#
actual_validation_numeric <- as.integer(
  validation.data$CARAVAN == "Yes"
)

rf_prediction <- ROCR::prediction(
  random_forest_validation_probability,
  actual_validation_numeric
)

rf_roc <- ROCR::performance(
  rf_prediction,
  measure = "tpr",
  x.measure = "fpr"
)

rf_auc <- ROCR::performance(
  rf_prediction,
  measure = "auc"
)

rf_auc_value <- as.numeric(
  rf_auc@y.values[[1]]
)

rf_auc_value

plot(
  rf_roc,
  main = paste(
    "Random Forest ROC Curve — AUC =",
    round(rf_auc_value, 3)
  ),
  xlab = "False Positive Rate (1 - Specificity)",
  ylab = "True Positive Rate (Recall)",
  lwd = 2
)

# Diagonal line represents random classification.
abline(
  a = 0,
  b = 1,
  lty = 2
)

selected_results <- evaluate_model(
  actual = validation.data$CARAVAN,
  probability = random_forest_validation_probability,
  threshold = random_forest_threshold
)

selected_recall <- selected_results$metrics$Recall

selected_false_positive_rate <- (
  1 - selected_results$metrics$Specificity
)
selected_false_positive_rate

plot(
  rf_roc,
  main = paste(
    "Random Forest ROC Curve — AUC =",
    round(rf_auc_value, 3)
  ),
  xlab = "False Positive Rate",
  ylab = "True Positive Rate",
  colorize = T,
  lwd = 2
)

abline(
  a = 0,
  b = 1,
  lty = 2
)

points(
  selected_false_positive_rate,
  selected_recall,
  pch = 19,
  cex = 1.5
)

text(
  selected_false_positive_rate,
  selected_recall,
  labels = paste0(
    " Threshold = ",
    random_forest_threshold
  ),
  pos = 4
)
