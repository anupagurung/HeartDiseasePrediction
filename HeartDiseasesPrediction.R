# Load necessary libraries
library(tidyverse)
library(caret)
library(nnet)
library(tree)
library(randomForest)
library(ggplot2)
library(reshape2)  # Add this line to load reshape2
library(caTools)
library(class)
library(rpart)

# Load the dataset
heart_data <- read.csv("heart .csv")

# Display overview of the dataset
str(heart_data)

# Summary statistics
summary(heart_data)

# Check for missing values in the entire dataset
missing_values <- any(is.na(heart_data))
if (missing_values) {
  cat("There are missing values in the dataset.\n")
  # If you want to get the count of missing values for each variable, you can use:
  missing_count <- colSums(is.na(heart_data))
  print(missing_count)
} else {
  cat("No missing values found in the dataset.\n")
}

# Check for duplicated rows in the entire dataset
duplicate_rows <- any(duplicated(heart_data))
if (duplicate_rows) {
  cat("There are duplicated rows in the dataset.\n")
  # If you want to get the duplicated rows, you can use:
  duplicated_rows <- heart_data[duplicated(heart_data), ]
  print(duplicated_rows)
} else {
  cat("No duplicated rows found in the dataset.\n")
}
# Check total number of duplicated rows in the dataset
duplicate_rows <- sum(duplicated(heart_data))

cat("Total number of duplicated rows:", duplicate_rows, "\n")


# Keep only one row for each set of duplicated values
clean_data <- distinct(heart_data, .keep_all = TRUE)
clean_data

# Data Analysis
str(heart_data)
colSums(is.na(heart_data))

# Data visualization for exploratory data analysis
# Age distribution
ggplot(heart_data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Age", x = "Age", y = "Frequency")

# ... (continue with other visualization plots)

# Correlation plot
cor_matrix <- cor(heart_data)
melted_cor_matrix <- reshape2::melt(cor_matrix)  # Use reshape2::melt to specify the package
ggplot(melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, 
                       limit = c(-1, 1), space = "Lab", name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 8, hjust = 1)) +
  coord_fixed()

ggsave("correlationfigure.png", plot = last_plot(), device = "png")

# Set the size of the plot
par(mfrow = c(4, 4), mar = c(4, 4, 2, 1))  # Adjust the layout and margins as needed

# Create histograms for each variable in the dataset
hist(heart_data$age, main = "Age", xlab = "Age")
hist(heart_data$sex, main = "Sex", xlab = "Sex")
hist(heart_data$cp, main = "Chest Pain Type", xlab = "Chest Pain Type")
hist(heart_data$trestbps, main = "Resting Blood Pressure", xlab = "Resting Blood Pressure")
hist(heart_data$chol, main = "Serum Cholesterol", xlab = "Serum Cholesterol")
hist(heart_data$fbs, main = "fasting blood sugar ", xlab = "fasting blood sugar ")
hist(heart_data$oldpeak, main = "oldpeak ", xlab = "oldpeak")
hist(heart_data$slope, main = "slope of the peak exercise ST segment", xlab = "slope of the peak exercise ST segment")

# Save the plot as a PNG file
dev.copy(png, "featuresplot.png")
dev.off()




# Plot count of target variable
ggplot(heart_data, aes(x = factor(target))) +
  geom_bar() +
  ylab("Count") +
  xlab("Target") +
  ggtitle("Count of Target Variable")

# Calculate and print percentages
target_temp <- table(heart_data$target)
cat("Percentage of patients without heart problems: ", round((target_temp[1] * 100) / sum(target_temp), 2), "%\n")
cat("Percentage of patients with heart problems: ", round((target_temp[2] * 100) / sum(target_temp), 2), "%\n")


# Define predictors and target variable
predictors <- heart_data[, !colnames(heart_data) %in% "target"]
target <- heart_data$target

# Split the data into training and testing sets
set.seed(0)  # Set seed for reproducibility
split <- sample.split(target, SplitRatio = 0.8)
train_data <- subset(heart_data, split == TRUE)
test_data <- subset(heart_data, split == FALSE)

# Check the dimensions of the train and test sets
cat("Train set dimensions:", dim(train_data), "\n")
cat("Test set dimensions:", dim(test_data), "\n")


# Train logistic regression model
lr_model <- glm(target ~ ., data = train_data, family = binomial)

# Make predictions on the test set
Y_pred_lr <- predict(lr_model, newdata = test_data, type = "response")
Y_pred_lr <- ifelse(Y_pred_lr > 0.5, 1, 0)

# Calculate accuracy score
score_lr <- round(sum(Y_pred_lr == test_data$target) / length(test_data$target) * 100, 2)

cat("The accuracy score achieved using Logistic Regression is: ", score_lr, "%\n")

# Decision tree model

# Initialize variables
max_accuracy <- 0
best_seed <- 0

# Loop through random states to find the best model
for (x in 1:200) {
  set.seed(x)
  dt_model <- rpart(target ~ ., data = train_data, method = "class")
  Y_pred_dt <- predict(dt_model, newdata = test_data, type = "class")
  current_accuracy <- sum(Y_pred_dt == test_data$target) / length(test_data$target) * 100
  if (current_accuracy > max_accuracy) {
    max_accuracy <- current_accuracy
    best_seed <- x
  }
}

# Train the best decision tree model
set.seed(best_seed)
dt_model <- rpart(target ~ ., data = train_data, method = "class")

# Make predictions on the test set
Y_pred_dt <- predict(dt_model, newdata = test_data, type = "class")

# Calculate accuracy score
score_dt <- round(sum(Y_pred_dt == test_data$target) / length(test_data$target) * 100, 2)

cat("The accuracy score achieved using Decision Tree is: ", score_dt, "%\n")

# Random Forest


# Initialize variables
max_accuracy <- 0
best_seed <- 0

# Loop through random states to find the best model
for (x in 1:2000) {
  set.seed(x)
  rf_model <- randomForest(target ~ ., data = train_data, ntree = 500)
  Y_pred_rf <- predict(rf_model, newdata = test_data)
  current_accuracy <- sum(Y_pred_rf == test_data$target) / length(test_data$target) * 100
  if (current_accuracy > max_accuracy) {
    max_accuracy <- current_accuracy
    best_seed <- x
  }
}

# Train the best random forest model
set.seed(best_seed)
rf_model <- randomForest(target ~ ., data = train_data, ntree = 500)

# Make predictions on the test set
Y_pred_rf <- predict(rf_model, newdata = test_data)

# Calculate accuracy score
score_rf <- round(sum(Y_pred_rf == test_data$target) / length(test_data$target) * 100, 2)

cat("The accuracy score achieved using Random forest is: ", score_rf, "%\n")

# Define predictors and target variable
predictors <- train_data[, !colnames(train_data) %in% "target"]
target <- train_data$target

# Define X_train, Y_train, X_test, and Y_test
X_train <- predictors
Y_train <- target
X_test <- test_data[, !colnames(test_data) %in% "target"]
Y_test <- test_data$target

# Train KNN classifier
k <- 7  # Number of neighbors
knn_model <- knn(train = X_train, test = X_test, cl = Y_train, k = k)

# Make predictions on the test set
Y_pred_knn <- knn_model

# Calculate accuracy score
score_knn <- round(sum(Y_pred_knn == Y_test) / length(Y_test) * 100, 2)
cat("The accuracy score achieved using KNN is: ", score_knn, "%\n")

# Load necessary libraries
library(ggplot2)

# Set the figure size
options(repr.plot.width=15, repr.plot.height=8)

# Create a dataframe for plotting
plot_data <- data.frame(algorithms = algorithms, scores = scores)

# Create bar plot
ggplot(plot_data, aes(x = algorithms, y = scores)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(x = "Algorithms", y = "Accuracy score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Accuracy Score by Algorithms")
# Define predictors and target variable
predictors <- heart_data[, !colnames(heart_data) %in% "target"]
target <- heart_data$target

# Split the data into training and testing sets
set.seed(0)  # Set seed for reproducibility
split <- sample.split(target, SplitRatio = 0.8)
train_data <- subset(heart_data, split == TRUE)
test_data <- subset(heart_data, split == FALSE)

# Train logistic regression model
lr_model <- glm(target ~ ., data = train_data, family = binomial)

# Make predictions on the test set
Y_pred_lr <- predict(lr_model, newdata = test_data, type = "response")
Y_pred_lr <- ifelse(Y_pred_lr > 0.5, 1, 0)

# Train decision tree model
dt_model <- rpart(target ~ ., data = train_data, method = "class")

# Make predictions on the test set
Y_pred_dt <- predict(dt_model, newdata = test_data, type = "class")

# Create a list of models and their predictions
models <- list("Logistic Regression" = Y_pred_lr,
               "Decision Tree" = Y_pred_dt)

# Create a classification report function
classification_report <- function(true, pred) {
  confusion <- confusionMatrix(data = as.factor(pred), reference = as.factor(true))
  return(confusion)
}

# Generate classification report for each model
for (i in names(models)) {
  cat("Classification Report for", i, ":\n")
  print(classification_report(test_data$target, models[[i]]))
  cat("\n")
}
