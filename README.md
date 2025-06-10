# Customer Churn Prediction

This repository contains a machine learning project focused on predicting customer churn for a telecommunications company. The project is implemented in a Jupyter Notebook (`client_charn_prediction.ipynb`) and uses a dataset to build and evaluate a predictive model. Below is an overview of the project structure, data, methodology, and results.

## Project Overview

The goal of this project is to develop a predictive model that determines the likelihood of customer churn based on various features such as demographic information, service usage, and billing details. The model is trained and evaluated using a dataset provided in CSV format, and the final predictions are submitted to Kaggle for scoring.

## Dataset

The dataset is split into three files, downloaded from Google Drive:
- `train.csv`: Training data with features and the target variable (`Churn`).
- `test.csv`: Test data for generating predictions.
- `submission.csv`: Template for submitting predictions to Kaggle.

### Features
The dataset includes the following features, categorized into numerical and categorical:
- **Numerical Features**:
    - `ClientPeriod`: Duration of the customer's subscription (in months).
    - `MonthlySpending`: Monthly payment amount.
    - `TotalSpent`: Total amount spent by the customer.
- **Categorical Features**:
    - `Sex`: Customer's gender (Male/Female).
    - `IsSeniorCitizen`: Whether the customer is a senior citizen (0/1).
    - `HasPartner`: Whether the customer has a partner (Yes/No).
    - `HasChild`: Whether the customer has children (Yes/No).
    - `HasPhoneService`: Whether the customer has phone service (Yes/No).
    - `HasMultiplePhoneNumbers`: Whether the customer has multiple phone numbers (Yes/No/No phone service).
    - `HasInternetService`: Type of internet service (DSL/Fiber optic/No).
    - `HasOnlineSecurityService`: Whether the customer has online security (Yes/No/No internet service).
    - `HasOnlineBackup`: Whether the customer has online backup (Yes/No/No internet service).
    - `HasDeviceProtection`: Whether the customer has device protection (Yes/No/No internet service).
    - `HasTechSupportAccess`: Whether the customer has tech support (Yes/No/No internet service).
    - `HasOnlineTV`: Whether the customer has online TV (Yes/No/No internet service).
    - `HasMovieSubscription`: Whether the customer has a movie subscription (Yes/No/No internet service).
    - `HasContractPhone`: Contract type (Month-to-month/One year/Two year).
    - `IsBillingPaperless`: Whether billing is paperless (Yes/No).
    - `PaymentMethod`: Payment method (Electronic check, Mailed check, Bank transfer, Credit card).

### Target Variable
- `Churn`: Binary variable indicating whether the customer churned (1) or not (0).

## Methodology

The project follows a structured approach to data analysis and model building:

1. **Data Loading and Exploration**:
    - The training and test datasets are loaded using `pandas`.
    - Initial exploration is performed using `data.head()` to inspect the data structure.
    - Features are categorized into numerical (`num_cols`) and categorical (`cat_cols`) for preprocessing.

2. **Data Preprocessing**:
    - Basic data cleaning is conducted to ensure data quality (specific cleaning steps are not detailed in the notebook but implied by the "Basic data clean" section).
    - Categorical features are identified for use in the CatBoost model.

3. **Model Training**:
    - A **CatBoostClassifier** is used as the primary model, leveraging its ability to handle categorical features natively.
    - Hyperparameter tuning is performed using `GridSearchCV` with 5-fold cross-validation to optimize the following parameters:
        - `iterations`: [100, 200, 400]
        - `learning_rate`: [0.01, 0.1, 0.3]
        - `depth`: [4, 7]
        - `l2_leaf_reg`: [3.0, 6.0]
    - A refined grid search is conducted with a narrower parameter range:
        - `iterations`: [90, 95, 100, 105, 110]
        - `learning_rate`: [0.09, 0.1, 0.11]
        - `depth`: [4]
        - `l2_leaf_reg`: [6.0]
    - The model is trained on a GPU for efficiency.

4. **Evaluation**:
    - The model's performance is evaluated using the **ROC-AUC** metric.
    - The best model achieves a ROC-AUC score of **0.8441** on the test set.

5. **Prediction and Submission**:
    - The best model is used to predict churn probabilities for the test dataset.
    - Predictions are formatted according to the Kaggle submission template and saved as `my_submission.csv`.

## Results

- The final model, after hyperparameter tuning, achieves a ROC-AUC score of **0.8441** on the test set.
- The submission file (`my_submission.csv`) contains predicted churn probabilities for the test dataset, ready for Kaggle evaluation.

## Dependencies

The project relies on the following Python libraries:
- `pandas`: For data manipulation and loading.
- `numpy`: For numerical operations.
- `matplotlib`: For plotting (though no plots are explicitly shown in the notebook).
- `gdown`: For downloading datasets from Google Drive.
- `catboost`: For the CatBoostClassifier model.
- `scikit-learn`: For `GridSearchCV` and evaluation metrics.

## File Structure

- `client_charn_prediction.ipynb`: Main Jupyter Notebook containing the project code.
- `train.csv`: Training dataset (downloaded during execution).
- `test.csv`: Test dataset (downloaded during execution).
- `submission.csv`: Submission template (downloaded during execution).
- `my_submission.csv`: Final submission file with predictions.

## How to Run

1. Clone this repository.
2. Install the required dependencies: `pip install pandas numpy matplotlib gdown catboost scikit-learn`.
3. Run the Jupyter Notebook (`client_charn_prediction.ipynb`) in an environment with Jupyter installed.
4. The notebook will download the datasets, train the model, and generate the submission file.
