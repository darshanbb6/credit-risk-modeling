# Credit Risk Modeling Using Machine Learning

## Project Overview
This project builds a machine learning classification model to predict credit risk using the German Credit dataset. The objective is to identify applicants labelled as having good or bad credit risk based on their financial and personal attributes.

Two Logistic Regression models are compared: a standard model and a class-balanced model.

## Objectives
- Explore and understand the credit dataset.
- Prepare categorical and numerical features for machine learning.
- Build a Logistic Regression classification pipeline.
- Evaluate model performance using multiple metrics.
- Compare standard and class-balanced models.

## Dataset
- **Name:** German Credit Data (Statlog)
- **Source:** UCI Machine Learning Repository
- **Dataset size:** 1,000 records
- **Features:** 20
- **Target:** Credit risk class

The original target labels are mapped as:
- `0` = Good credit
- `1` = Bad credit

The dataset contains 700 good-credit and 300 bad-credit records.

## Tools and Technologies
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- VS Code / Jupyter Notebook

## Methodology
1. Loaded and inspected the dataset.
2. Checked for missing values and duplicate records.
3. Separated features and target labels.
4. Split the data into training (80%) and testing (20%) sets using stratification.
5. Imputed missing values through preprocessing pipelines.
6. Scaled numerical features and one-hot encoded categorical features.
7. Trained standard and class-balanced Logistic Regression models.
8. Evaluated both models using accuracy, precision, recall, F1-score, and confusion matrices.

## Model Performance

| Metric | Original Model | Balanced Model |
|---|---:|---:|
| Accuracy | 78.00% | 75.00% |
| Precision | 66.67% | 55.81% |
| Recall | 53.33% | 80.00% |
| F1-score | 59.26% | 65.75% |

The balanced model achieved higher recall for the bad-credit class, identifying 48 of 60 actual bad-credit cases in the test set, compared with 32 of 60 for the original model.

However, the balanced model also incorrectly classified more good-credit cases as bad credit. This illustrates the trade-off between detecting risky cases and avoiding false alarms.

## Visualizations

### Original Logistic Regression
![Original model confusion matrix](images/original_confusion_matrix.png)

### Balanced Logistic Regression
![Balanced model confusion matrix](images/balanced_confusion_matrix.png)

## Key Learnings
- Why accuracy alone is insufficient for evaluating classification models.
- How preprocessing pipelines handle numerical and categorical features.
- How class weighting affects precision and recall.
- How confusion matrices reveal different types of prediction errors.

## Limitations and Responsible Use
This is an educational machine learning project, not a production-ready credit scoring system. The dataset is limited in size and may not represent current applicants or lending conditions. The model can make errors and may reflect biases in historical data.

Attributes such as age, sex-related information, and foreign-worker status raise fairness and legal considerations. Further validation, fairness testing, and compliance review would be necessary before any real-world use. Predictions should not be used alone to make lending decisions.

## Project Files
- `README.md` — Project documentation
- `model_comparison.csv` — Evaluation metrics for both models
- `images/original_confusion_matrix.png` — Original model evaluation
- `images/balanced_confusion_matrix.png` — Balanced model evaluation
- Python notebook or script — Data preparation, training, and evaluation

## Conclusion
This project demonstrates an end-to-end introductory credit risk classification workflow using Python and Scikit-learn. Comparing standard and class-balanced Logistic Regression models highlights how the choice of evaluation metric affects model assessment.
