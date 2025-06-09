💧 Pump It Up: Predicting Water Pump Functionality
This project is part of a data science competition hosted by DrivenData. The goal is to predict the functional status of water pumps across Tanzania using machine learning techniques. The classification has three possible outcomes: functional, functional needs repair, and non functional.

📂 Dataset
Train set: 59,400 observations with labels

Test set: 14,850 observations (without labels)

Target variable: status_group

Features: Geographical, operational, and technical information about waterpoints

📊 Project Workflow
1. Exploratory Data Analysis (EDA)
Checked dimensions and variable types (str(), dim())

Identified and summarized missing values using summarise_all() and pivot_longer()

Removed columns with excessive missingness (e.g., scheme_name)

Imputed missing values using:

"unknown" for categorical columns like installer, funder

Mode for binary columns such as permit, public_meeting

2. Feature Engineering
Extracted year, month, and day from date_recorded

Created well_age feature from construction_year and year_recorded

Replaced 0 construction years with NA and imputed using median age

3. Data Preparation
Converted character columns to factors

Removed near-zero variance features using nearZeroVar()

Removed highly correlated numeric features using correlation matrix

Dropped high-cardinality categorical features (e.g., subvillage, funder, ward)

🤖 Models
✅ Random Forest (Final Model)
Trained using the randomForest package (ntree = 100)

Achieved 93.8% accuracy on the training set

Evaluated with sensitivity, specificity, and balanced accuracy for all classes

Feature importance analyzed using varImpPlot()

✅ Logistic Regression (Baseline)
Used nnet::multinom for multiclass logistic regression

Faster training but significantly lower accuracy (~73.8%)

Unable to capture non-linear relationships as effectively as tree-based models

✅ Cross-Validation
5-fold cross-validation using caret::train() with method = "ranger"

Confirmed model’s generalizability and robustness

🧪 Test Set Predictions
The trained Random Forest model was applied to the test set using predict()

Test data was preprocessed identically to training data

Predictions were saved into a submission.csv file:

python-repl
Kopyala
Düzenle
id,status_group
50785,functional
51630,non functional
...
🏁 Conclusion
Random Forest proved to be the most effective model for this task due to its ability to handle complex, non-linear patterns and categorical data. The project followed a complete data science pipeline: from EDA and cleaning to model evaluation and deployment-ready predictions.

📁 Files
training_set_values.csv

training_set_labels.csv

test_set_values.csv

submission.csv

project_script.R (contains full code)

✨ Tools & Libraries
R, tidyverse, caret, randomForest, ranger, nnet, doParallel

