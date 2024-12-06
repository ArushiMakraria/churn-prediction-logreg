# Customer Churn Analysis using Logistic Regression

## Overview
This project addresses customer churn prediction for a telecommunications company using data from `Telco-Customer-Churn.xlsx`. The goal is to preemptively identify customers likely to leave the company and provide actionable insights to improve retention strategies. The analysis employs binary logistic regression with various evaluation metrics to assess model performance and recommend optimal classification thresholds.

## Objectives
1. Understand the relationship between predictors and churn.
2. Train a binary logistic regression model using the All-Possible Subsets method to select the best-fit model based on the Bayesian Information Criterion (BIC).
3. Evaluate the model's goodness-of-fit and prediction capabilities.
4. Recommend an optimal probability threshold for churn classification based on performance metrics.

## Dataset Details
- **Response Variable:**
  - `Churn` (Event category: `Yes`)
- **Categorical Predictors:**
  - `Contract`, `Dependents`, `Gender`, `InternetService`, `MultipleLines`, `PaperlessBilling`, `Partner`, `PhoneService`, and `SeniorCitizen`
  - Categories reordered in ascending order of frequency.
- **Interval Predictors:**
  - `MonthlyCharges`, `Tenure`, `TotalCharges`

## Methodology
### Exploratory Analysis
1. **Categorical Predictors:**
   - Generated vertical bar charts showing the odds of churn for each category.
   - Displayed categories in descending order of churn odds with a reference line for overall churn odds.
   - Interpreted the potential impact of each predictor on churn.
2. **Interval Predictors:**
   - Generated horizontal boxplots grouped by churn categories with a reference line for the overall mean.
   - Interpreted the potential impact of each predictor on churn.

### Model Training
- **Specifications:**
  - Distribution: Bernoulli
  - Link Function: Logit
  - Selection Method: All-Possible Subsets (Model chosen based on lowest BIC value).
  - Missing values for predictors and target variable were dropped.
- **Output:**
  - CSV summarizing model candidates with non-aliased parameters, log-likelihood, AIC, and BIC values sorted by ascending BIC.
  - Predicted probability of churn for a sample customer profile.

### Model Evaluation
1. **Goodness-of-Fit:**
   - Calculated and visualized residuals (Simple, Pearson, Deviance) by observed churn status.
   - Assessed areas of poor model fit.
2. **Performance Metrics:**
   - McFadden’s R-squared
   - Cox-Snell’s R-squared
   - Nagelkerke’s R-squared
   - Tjur’s Coefficient of Discrimination
   - Area Under Curve (AUC) for acceptability evaluation.
   - Root Average Squared Error (RASE).

### Classification Threshold
1. **Kolmogorov-Smirnov Chart:**
   - Identified KS statistic and corresponding probability threshold.
   - Evaluated misclassification rate using this threshold.
2. **Precision-Recall Chart:**
   - Identified F1 score and corresponding probability threshold.
   - Evaluated misclassification rate using this threshold.

## Results
- **Final Model:** Selected based on lowest BIC value.
- **Key Metrics:**
  - Predicted churn probabilities for customer profiles.
  - Residual analysis revealed model fit and potential areas for improvement.
  - Evaluated performance metrics confirmed model acceptability.
- **Recommended Thresholds:** Identified optimal thresholds for churn classification based on KS statistic and F1 score.

## How to Use This Repository
1. Clone the repository:
   ```bash
   git clone [<repository_url>](https://github.com/ArushiMakraria/churn-prediction-logreg)
   ```
2. Access the analysis scripts and dataset.
3. Follow the provided notebooks for step-by-step implementation and insights.

For questions or feedback, feel free to reach out!
