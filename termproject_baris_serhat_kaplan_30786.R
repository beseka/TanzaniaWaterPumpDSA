install.packages(c("tidyverse", "data.table", "caret", "randomForest","ranger", "doParallel","nnet"))
library(tidyverse)
library(data.table)
library(caret)
library(randomForest)
library(readr)
library(dplyr)
library(ranger)
library(doParallel)
library(nnet)

train_features <- read_csv("training_set_values.csv")
train_labels <- read_csv("trainin_set_labels.csv")
test_features <- read_csv("test_set_values.csv")

# Data analysis part

head(train_features)
head(train_labels)

# dimensions
dim(train_features)
dim(train_labels)
dim(test_features)

# appopriate testing for data types 
str(train_features)
str(train_labels)

# null/na information (train set)
missing_train <- train_features %>%
  summarise_all(~sum(is.na(.))) %>%
  pivot_longer(cols = everything(), names_to = "column", values_to = "missing_count") %>%
  mutate(missing_ratio = missing_count / nrow(train_features)) %>%
  arrange(desc(missing_count))

# (test set)
missing_test <- test_features %>%
  summarise_all(~sum(is.na(.))) %>%
  pivot_longer(cols = everything(), names_to = "column", values_to = "missing_count") %>%
  mutate(missing_ratio = missing_count / nrow(test_features)) %>%
  arrange(desc(missing_count))

head(missing_train)
head(missing_test)

#scheme_name has mostly missing values D1 will dropping it 
#scheme_management, installer and funder D1 will going to fill it with unknown 
#permit and public_meeting's are binary desicions D1 will going to fill with mode
#these are for avoiding overfitting

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

test_features  <- test_features  %>% select(-scheme_name)
train_features <- train_features %>% select(-scheme_name)

fill_unknown_cols <- c("scheme_management", "installer", "funder")

train_features[fill_unknown_cols] <- lapply(train_features[fill_unknown_cols], function(x) ifelse(is.na(x), "unknown", x))
test_features[fill_unknown_cols]  <- lapply(test_features[fill_unknown_cols], function(x) ifelse(is.na(x), "unknown", x))

most_common_public_meeting <- Mode(na.omit(train_features$public_meeting))
most_common_permit <- Mode(na.omit(train_features$permit))

train_features$public_meeting[is.na(train_features$public_meeting)] <- most_common_public_meeting
test_features$public_meeting[is.na(test_features$public_meeting)] <- most_common_public_meeting

train_features$permit[is.na(train_features$permit)] <- most_common_permit
test_features$permit[is.na(test_features$permit)] <- most_common_permit


#control part for data preprocessing

sum(is.na(train_features))
sum(is.na(test_features))
#these are not 0

na_train_details <- colSums(is.na(train_features)) %>%
  sort(decreasing = TRUE)
na_train_details[na_train_details > 0]
na_test_details <- colSums(is.na(test_features)) %>%
  sort(decreasing = TRUE)
na_test_details[na_test_details > 0]

#subvillage have null values 
train_features$subvillage[is.na(train_features$subvillage)] <- "unknown"
test_features$subvillage[is.na(test_features$subvillage)] <- "unknown"

sum(is.na(train_features))
sum(is.na(test_features))
#now these are 0 

colSums(is.na(train_features)) %>% sort(decreasing = TRUE) %>% head(10)
colSums(is.na(test_features)) %>% sort(decreasing = TRUE) %>% head(10)


#training data 
train_data <- left_join(train_features, train_labels, by = "id")

dim(train_data)

table(train_data$status_group)


# Feature engineering

train_data <- train_data %>%
  mutate(
    year_recorded = lubridate::year(date_recorded),
    month_recorded = lubridate::month(date_recorded),
    day_recorded = lubridate::day(date_recorded)
  )

test_features <- test_features %>%
  mutate(
    year_recorded = lubridate::year(date_recorded),
    month_recorded = lubridate::month(date_recorded),
    day_recorded = lubridate::day(date_recorded)
  )

train_data <- train_data %>%
  mutate(
    construction_year = ifelse(construction_year == 0, NA, construction_year),
    well_age = year_recorded - construction_year
  )

test_features <- test_features %>%
  mutate(
    construction_year = ifelse(construction_year == 0, NA, construction_year),
    well_age = year_recorded - construction_year
  )

median_well_age <- median(train_data$well_age, na.rm = TRUE)

train_data$well_age[is.na(train_data$well_age)] <- median_well_age
test_features$well_age[is.na(test_features$well_age)] <- median_well_age

str(train_data)
summary(train_data$well_age)

#categorical features to factor

char_cols <- sapply(train_data, is.character)
cat_cols <- names(train_data)[char_cols]
cat_cols <- setdiff(cat_cols, "status_group")  # not label
cat_cols

train_data[cat_cols] <- lapply(train_data[cat_cols], as.factor)

test_features[cat_cols] <- lapply(test_features[cat_cols], as.factor)

train_data$status_group <- as.factor(train_data$status_group)

str(train_data$status_group)  # 3 level: functional, non functional, needs repair
summary(train_data$status_group)

#feature selection
zero_var_cols <- nearZeroVar(train_data, names = TRUE, freqCut = 99, uniqueCut = 1)

train_data <- train_data %>% select(-all_of(zero_var_cols))
test_features <- test_features %>% select(-all_of(zero_var_cols))

median_year <- median(train_data$construction_year, na.rm = TRUE)
train_data$construction_year[is.na(train_data$construction_year)] <- median_year

train_data <- train_data %>%
  mutate(well_age = year_recorded - construction_year)

colSums(is.na(train_data)) %>% sort(decreasing = TRUE) %>% head(10)

numeric_cols <- sapply(train_data, is.numeric)
numeric_data <- train_data[, numeric_cols]

cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")

high_corr <- findCorrelation(cor_matrix, cutoff = 0.9, names = TRUE, verbose = TRUE)

print(high_corr)

train_data <- train_data %>% select(-all_of(high_corr))
test_features <- test_features %>% select(-all_of(high_corr))

cat_cols <- sapply(train_data, is.factor)
sapply(train_data[, cat_cols], nlevels) %>% sort(decreasing = TRUE)
high_cardinality_cols <- c("wpt_name", "subvillage", "installer", "ward", "funder","lga")

train_data <- train_data %>% select(-all_of(high_cardinality_cols))
test_features <- test_features %>% select(-all_of(high_cardinality_cols))


#random forest model
set.seed(42)
rf_model <- randomForest(status_group ~ ., data = train_data %>% select(-id), ntree = 100, importance = TRUE)

varImpPlot(rf_model, type = 2)


train_pred <- predict(rf_model, newdata = train_data)

confusionMatrix(train_pred, train_data$status_group)
confusionMatrix(train_pred, train_data$status_group)$overall["Accuracy"]

#logistic regression
train_data$status_group <- as.factor(train_data$status_group)

log_model <- multinom(status_group ~ ., data = train_data %>% select(-id))

log_preds <- predict(log_model, newdata = train_data)

confusionMatrix(log_preds, train_data$status_group)$overall["Accuracy"]

#cross validation

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

ctrl <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

set.seed(123)
cv_model <- train(
  status_group ~ .,
  data = train_data %>% select(-id),
  method = "ranger",
  trControl = ctrl,
  importance = "impurity"
)

stopCluster(cl)
registerDoSEQ()

cv_model

# Predictions on test_set
test_features$construction_year[is.na(test_features$construction_year)] <- median_year

test_predictions <- predict(rf_model, newdata = test_features)
submission <- data.frame(
  id = test_features$id,
  status_group = test_predictions
)

write_csv(submission, "submission.csv")

getwd()

