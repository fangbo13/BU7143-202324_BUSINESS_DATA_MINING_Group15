library(dplyr)
library(readr)
library(readxl)
library(ggplot2)
library(zoo)
library(forecast)
library(e1071)
library(fastDummies)
library(corrplot)
library(caret)
library(class)
library(rpart)
library(rpart.plot)
library(randomForest)
library(neuralnet)
library(xgboost)
library(pROC)

airline <- read_csv("C:/Users/48869/OneDrive/桌面/Business analytics/Data Mining/group/Invistico_Airline.csv")
View(airline)

#====================================check missing_values=======================================
missing_values <- is.na(airline)
missing_counts <- colSums(missing_values)
missing_counts

# clean missing value row
airline_clean <- na.omit(airline)
View(airline_clean)
#==============================================EDA==============================================
variable<- c("satisfaction", "Customer Type", "Type of Travel", "Class")
for (i in variable) {
  print(paste(i, "\n"))
  print(table(airline_clean[[i]]))
}
names(airline_clean) <- gsub(" ", "_", names(airline_clean))

#============================
# Satisfaction
ggplot(airline_clean, aes(x = satisfaction)) +
  geom_bar(fill = "blue", alpha = 0.8) +
  labs(title = "Satisfaction Distribution", x = "Satisfaction", y = "Count") +
  theme_minimal()

# Customer Type
ggplot(airline_clean, aes(x = Customer_Type)) +
  geom_bar(fill = "red", alpha = 0.8) +
  labs(title = "Customer Type Distribution", x = "Customer Type", y = "Count") +
  theme_minimal()

# Type of Travel
ggplot(airline_clean, aes(x = Type_of_Travel)) +
  geom_bar(fill = "green", alpha = 0.8) +
  labs(title = "Type of Travel Distribution", x = "Type of Travel", y = "Count") +
  theme_minimal()

# Class
ggplot(airline_clean, aes(x = Class)) +
  geom_bar(fill = "steelblue", alpha = 0.8) +
  labs(title = "Class Distribution", x = "Class", y = "Count") +
  theme_minimal()
#============================================

numeric_vars <- airline_clean %>%
  select(where(is.numeric))

cor_matrix <- cor(numeric_vars, use = "complete.obs")

corrplot(cor_matrix, method = "color", type = "upper", order = "hclust",
         tl.col = "black", tl.srt = 45, addCoef.col = "black", number.cex = 0.7,
         tl.cex = 0.7, cl.cex = 0.7, addCoefasPercent = TRUE)

#=========================================Drop lower impact  columns===========================================
set.seed(15)

categorical_vars <- c("satisfaction", "Customer_Type", "Type_of_Travel", "Class")
airline_clean <- dummy_cols(airline_clean, select_columns = categorical_vars,
                            remove_first_dummy = TRUE, remove_selected_columns = TRUE)

airline_reduced <- airline_clean %>%
  select(-Flight_Distance,
         -`Departure/Arrival_time_convenient`,
         -Gate_location,
         -Departure_Delay_in_Minutes,
         -Arrival_Delay_in_Minutes)
#======================================split data set ======================================================

norm_df <- airline_reduced


set.seed(15)

train_index <- sample(1:nrow(norm_df), 0.8 * nrow(norm_df))
valid_index <- setdiff(1:nrow(norm_df), train_index)

train_df <- norm_df[train_index, ]
valid_df <- norm_df[valid_index, ]


norm_values <- preProcess(train_df, method = "range")

train_norm_df <- predict(norm_values, train_df)
valid_norm_df <- predict(norm_values, valid_df)


cor_matrix <- cor(train_norm_df)
print(round(cor_matrix, 2))

color_palette <- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(x = cor_matrix, col = color_palette, symm = TRUE)

View(train_norm_df)

#==============================================model creation ======================================

#===============================================logistic_model==========================================
logistic_model <- glm(satisfaction_satisfied ~ ., data = train_norm_df, family = binomial)


summary(logistic_model)


valid_pred_prob <- predict(logistic_model, valid_norm_df, type = "response")
valid_pred_class <- ifelse(valid_pred_prob > 0.5, 1, 0)


confusion_matrix <- table(Predicted = valid_pred_class, Actual = valid_norm_df$satisfaction_satisfied)
confusion_matrix

#Accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))

confusion_logistic <- confusionMatrix(as.factor(valid_pred_class), as.factor(valid_norm_df$satisfaction_satisfied))


print(confusion_logistic)


sensitivity_logistic <- confusion_logistic$byClass['Sensitivity']
specificity_logistic <- confusion_logistic$byClass['Specificity']

print(paste("Sensitivity for Logistic Regression:", round(sensitivity_logistic, 2)))
print(paste("Specificity for Logistic Regression:", round(specificity_logistic, 2)))

confusion_data_logistic <- as.data.frame(confusion_logistic$table)

ggplot(data = confusion_data_logistic, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix for Logistic Regression Model", x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))



confusion <- confusionMatrix(as.factor(valid_pred_class), as.factor(valid_norm_df$satisfaction_satisfied))
print(confusion$byClass)

#========================================================================


predictions <- predict(logistic_model, newdata = valid_norm_df, type = "response")
actuals <- valid_norm_df$satisfaction_satisfied
logistic_rmse <- RMSE(predictions, actuals)
print(paste("RMSE for logistic model:", logistic_rmse))

#===============================================KNN ==============================================
k <- 5
valid_pred_knn <- knn(train = train_norm_df[,-which(names(train_norm_df) == "satisfaction_satisfied")],
                      test = valid_norm_df[,-which(names(valid_norm_df) == "satisfaction_satisfied")],
                      cl = train_norm_df$satisfaction_satisfied,
                      k = k)

confusion_matrix_knn <- table(Predicted = valid_pred_knn, Actual = valid_norm_df$satisfaction_satisfied)
print(confusion_matrix_knn)

accuracy_knn <- sum(diag(confusion_matrix_knn)) / sum(confusion_matrix_knn)
print(paste("Accuracy for KNN:", accuracy_knn))

valid_pred_knn_num <- as.numeric(levels(valid_pred_knn))[valid_pred_knn]
rmse_knn <- RMSE(valid_pred_knn_num, valid_norm_df$satisfaction_satisfied)
print(paste("RMSE for KNN model:", rmse_knn))



confusion_knn <- confusionMatrix(as.factor(valid_pred_knn), as.factor(valid_norm_df$satisfaction_satisfied))


print(confusion_knn)

sensitivity_knn <- confusion_knn$byClass["Sensitivity"]
specificity_knn <- confusion_knn$byClass["Specificity"]
print(paste("Sensitivity for KNN:", round(sensitivity_knn, 2)))
print(paste("Specificity for KNN:", round(specificity_knn, 2)))


confusion_data_knn <- as.data.frame(confusion_knn$table)

ggplot(data = confusion_data_knn, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix for KNN Model", x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

#==============================================Decision Tree ==========================================
# names(valid_norm_df) <- gsub(" ", "_", names(valid_norm_df))
# names(valid_norm_df) <- gsub("-", "_", names(valid_norm_df))
# View(valid_norm_df)

formula_dt <- satisfaction_satisfied ~ .

decision_tree_model <- rpart(formula_dt, data = train_norm_df, method = "class")
print(summary(decision_tree_model))

rpart.plot(decision_tree_model, type = 0, extra = 106, under = TRUE, cex = 0.6)
valid_pred_dt <- predict(decision_tree_model, valid_norm_df, type = "class")

dt_accuracy <- sum(valid_pred_dt == valid_norm_df$satisfaction_satisfied) / nrow(valid_norm_df)
print(paste("Decision Tree Accuracy:", dt_accuracy))

conf_matrix_dt <- confusionMatrix(as.factor(valid_pred_dt), as.factor(valid_norm_df$satisfaction_satisfied))
print(conf_matrix_dt)

sensitivity_dt <- conf_matrix_dt$byClass["Sensitivity"]
specificity_dt <- conf_matrix_dt$byClass["Specificity"]

print(paste("Sensitivity for Decision Tree:", round(sensitivity_dt, 2)))
print(paste("Specificity for Decision Tree:", round(specificity_dt, 2)))

confusion_data_dt <- as.data.frame(conf_matrix_dt$table)

ggplot(data = confusion_data_dt, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix for Decision Tree Model", x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


#===============================Random forest ==========================================
names(train_norm_df) <- gsub(" ", "_", names(train_norm_df))
names(train_norm_df) <- gsub("-", "_", names(train_norm_df))
names(valid_norm_df) <- gsub(" ", "_", names(valid_norm_df))
names(valid_norm_df) <- gsub("-", "_", names(valid_norm_df))

train_norm_df$satisfaction_satisfied <- as.factor(train_norm_df$satisfaction_satisfied)
valid_norm_df$satisfaction_satisfied <- as.factor(valid_norm_df$satisfaction_satisfied)


formula_rf <- satisfaction_satisfied ~ .


set.seed(15)
random_forest_model <- randomForest(formula_rf, data = train_norm_df, ntree = 200, mtry = floor(sqrt(ncol(train_norm_df)-1)), importance = TRUE)
print(summary(random_forest_model))


importance(random_forest_model)
varImpPlot(random_forest_model)


rf_predictions <- predict(random_forest_model, valid_norm_df)


rf_confusion_matrix <- confusionMatrix(rf_predictions, valid_norm_df$satisfaction_satisfied)
print(rf_confusion_matrix)


rf_accuracy <- sum(diag(rf_confusion_matrix$table)) / sum(rf_confusion_matrix$table)
print(paste("Random Forest Accuracy:", rf_accuracy))


rf_sensitivity <- rf_confusion_matrix$byClass["Sensitivity"]
rf_specificity <- rf_confusion_matrix$byClass["Specificity"]
print(paste("Random Forest Sensitivity:", round(rf_sensitivity, 2)))
print(paste("Random Forest Specificity:", round(rf_specificity, 2)))

rf_confusion_data <- as.data.frame(rf_confusion_matrix$table)
ggplot(data = rf_confusion_data, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix for Random Forest Model", x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


#===============================Neuronal net ==========================================

# train_norm_df$satisfaction_satisfied <- as.factor(train_norm_df$satisfaction_satisfied)
# valid_norm_df$satisfaction_satisfied <- as.factor(valid_norm_df$satisfaction_satisfied)
#
# train_norm_df$satisfaction_satisfied <- as.numeric(train_norm_df$satisfaction_satisfied) - 1
# valid_norm_df$satisfaction_satisfied <- as.numeric(valid_norm_df$satisfaction_satisfied) - 1
#
# nn_formula <- satisfaction_satisfied ~ .
#
# nn_model <- neuralnet(nn_formula, data = train_norm_df, hidden = c(5), linear.output = FALSE)
#
#
# plot(nn_model)
#
# nn_predictions <- compute(nn_model, valid_norm_df[, -which(names(valid_norm_df) == "satisfaction_satisfied")])
# nn_pred_values <- nn_predictions$net.result
#
# nn_pred_class <- ifelse(nn_pred_values > 0.5, 1, 0)
#
# nn_confusion_matrix <- confusionMatrix(as.factor(nn_pred_class), as.factor(valid_norm_df$satisfaction_satisfied))
# print(nn_confusion_matrix)
#
# nn_accuracy <- sum(diag(nn_confusion_matrix$table)) / sum(nn_confusion_matrix$table)
# print(paste("Neural Network Accuracy:", nn_accuracy))
#
# nn_sensitivity <- nn_confusion_matrix$byClass["Sensitivity"]
# nn_specificity <- nn_confusion_matrix$byClass["Specificity"]
# print(paste("Neural Network Sensitivity:", round(nn_sensitivity, 2)))
# print(paste("Neural Network Specificity:", round(nn_specificity, 2)))
#
# nn_confusion_data <- as.data.frame(nn_confusion_matrix$table)
# ggplot(data = nn_confusion_data, aes(x = Reference, y = Prediction, fill = Freq)) +
#   geom_tile(color = "white") +
#   geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
#   scale_fill_gradient(low = "white", high = "steelblue") +
#   labs(title = "Confusion Matrix for Neural Network Model", x = "Actual", y = "Predicted") +
#   theme_minimal() +
#   theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))



#===============================XGB  ==========================================
dtrain <- xgb.DMatrix(data = as.matrix(train_norm_df[-which(names(train_norm_df) == "satisfaction_satisfied")]), label = as.numeric(train_norm_df$satisfaction_satisfied) - 1)
dvalid <- xgb.DMatrix(data = as.matrix(valid_norm_df[-which(names(valid_norm_df) == "satisfaction_satisfied")]), label = as.numeric(valid_norm_df$satisfaction_satisfied) - 1)


params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.3,
  nthread = 2
)


set.seed(15)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, watchlist = list(val=dvalid), verbose = 0)


xgb_predictions <- predict(xgb_model, dvalid)
xgb_pred_class <- ifelse(xgb_predictions > 0.5, 1, 0)

xgb_confusion_matrix <- confusionMatrix(as.factor(xgb_pred_class), as.factor(valid_norm_df$satisfaction_satisfied))
print(xgb_confusion_matrix)

xgb_accuracy <- sum(diag(xgb_confusion_matrix$table)) / sum(xgb_confusion_matrix$table)
print(paste("XGBoost Accuracy:", xgb_accuracy))

xgb_sensitivity <- xgb_confusion_matrix$byClass["Sensitivity"]
xgb_specificity <- xgb_confusion_matrix$byClass["Specificity"]
print(paste("XGBoost Sensitivity:", round(xgb_sensitivity, 2)))
print(paste("XGBoost Specificity:", round(xgb_specificity, 2)))

xgb_confusion_data <- as.data.frame(xgb_confusion_matrix$table)
ggplot(data = xgb_confusion_data, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%0.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix for XGBoost Model", x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

#===============================ACC  ==========================================
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
accuracy

accuracy_knn <- sum(diag(confusion_matrix_knn)) / sum(confusion_matrix_knn)
accuracy_knn

dt_accuracy <- sum(valid_pred_dt == valid_norm_df$satisfaction_satisfied) / nrow(valid_norm_df)
dt_accuracy

rf_accuracy <- sum(diag(rf_confusion_matrix$table)) / sum(rf_confusion_matrix$table)
rf_accuracy

xgb_accuracy <- sum(diag(xgb_confusion_matrix$table)) / sum(xgb_confusion_matrix$table)
xgb_accuracy
##=============================== AUC and ROC  ==========================================
roc_logistic <- roc(valid_norm_df$satisfaction_satisfied, valid_pred_prob)
roc_knn <- roc(valid_norm_df$satisfaction_satisfied, valid_pred_knn_num)
roc_dt <- roc(valid_norm_df$satisfaction_satisfied, as.numeric(valid_pred_dt))
roc_rf <- roc(valid_norm_df$satisfaction_satisfied, as.numeric(rf_predictions))
roc_xgb <- roc(valid_norm_df$satisfaction_satisfied, xgb_predictions)

ggroc(list(logistic = roc_logistic, knn = roc_knn, decision_tree = roc_dt, random_forest = roc_rf, xgboost = roc_xgb)) +
  labs(title = "ROC Curves for Various Models", x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal() +
  theme(legend.position = "bottom")


auc_logistic <- auc(roc_logistic)
auc_knn <- auc(roc_knn)
auc_dt <- auc(roc_dt)
auc_rf <- auc(roc_rf)
auc_xgb <- auc(roc_xgb)

print(paste("AUC for Logistic Regression:", auc_logistic))
print(paste("AUC for KNN:", auc_knn))
print(paste("AUC for Decision Tree:", auc_dt))
print(paste("AUC for Random Forest:", auc_rf))
print(paste("AUC for XGBoost:", auc_xgb))

##=============================== feature importance ==========================================

importance_matrix <- xgb.importance(feature_names = colnames(train_norm_df[-which(names(train_norm_df) == "satisfaction_satisfied")]), model = xgb_model)

print(importance_matrix)

xgb.plot.importance(importance_matrix)
##=============================== feature  ==========================================
#F1
calculate_metrics <- function(actual, predicted) {
  cm <- confusionMatrix(as.factor(predicted), as.factor(actual))
  sensitivity <- cm$byClass["Sensitivity"]
  ppv <- cm$byClass["Pos Pred Value"]
  f1 <- ifelse(is.na(sensitivity) || is.na(ppv) || (sensitivity + ppv) == 0, NA, 2 * sensitivity * ppv / (sensitivity + ppv))
  return(list(f1 = f1, recall = sensitivity, precision = ppv))
}


metrics_logistic <- calculate_metrics(valid_norm_df$satisfaction_satisfied, valid_pred_class)
print(paste("Logistic Regression - F1 Score:", metrics_logistic$f1))

metrics_knn <- calculate_metrics(valid_norm_df$satisfaction_satisfied, as.factor(valid_pred_knn))
print(paste("KNN - F1 Score:", metrics_knn$f1))

metrics_dt <- calculate_metrics(valid_norm_df$satisfaction_satisfied, as.factor(valid_pred_dt))
print(paste("Decision Tree - F1 Score:", metrics_dt$f1))

metrics_rf <- calculate_metrics(valid_norm_df$satisfaction_satisfied, as.factor(rf_predictions))
print(paste("Random Forest - F1 Score:", metrics_rf$f1))

metrics_xgb <- calculate_metrics(valid_norm_df$satisfaction_satisfied, as.factor(xgb_pred_class))
print(paste("XGBoost - F1 Score:", metrics_xgb$f1))

