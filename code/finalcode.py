import pandas as pd
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations
import time
import Regression  # Assuming this module contains the necessary functions
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

# Load the data
data = pd.read_excel("Telco-Customer-Churn.xlsx")

# Drop rows with missing values
data.dropna(subset=['Churn', 'Contract', 'Dependents', 'Gender', 'InternetService', 
                    'MultipleLines', 'PaperlessBilling', 'Partner', 'PhoneService', 
                    'SeniorCitizen', 'MonthlyCharges', 'Tenure', 'TotalCharges'], inplace=True)

# Calculate overall odds of churn
overall_churn_odds = data['Churn'].value_counts(normalize=True)['Yes'] / data['Churn'].value_counts(normalize=True)['No']

# Define categorical predictors
categorical_predictors = ['Contract', 'Dependents', 'Gender', 'InternetService', 
                          'MultipleLines', 'PaperlessBilling', 'Partner', 'PhoneService', 
                          'SeniorCitizen']

# Function to calculate odds of churn for each category
def calculate_churn_odds_by_category(data, predictor):
    churn_odds_by_category = {}
    categories = data[predictor].unique()
    for category in categories:
        churn_odds_by_category[category] = data[data[predictor] == category]['Churn'].value_counts(normalize=True).get('Yes', 0) / data[data[predictor] == category]['Churn'].value_counts(normalize=True).get('No', 1)
    return churn_odds_by_category

# Function to plot odds of churn for each category
def plot_churn_odds_by_category(churn_odds_by_category, overall_churn_odds, predictor):
    sorted_categories = sorted(churn_odds_by_category, key=churn_odds_by_category.get, reverse=True)
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(churn_odds_by_category.keys()), y=list(churn_odds_by_category.values()), order=sorted_categories)
    plt.axhline(y=overall_churn_odds, color='r', linestyle='--', label='Overall Churn Odds')
    plt.title(f'Odds of Churn by {predictor}')
    plt.xlabel(predictor)
    plt.ylabel('Odds of Churn')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()

# Generate plots for each categorical predictor
for predictor in categorical_predictors:
    churn_odds_by_category = calculate_churn_odds_by_category(data, predictor)
    plot_churn_odds_by_category(churn_odds_by_category, overall_churn_odds, predictor)


# Function to plot boxplots for each interval predictor
def plot_boxplots_by_target(data, predictor):
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Churn', y=predictor, data=data)
    overall_mean = data[predictor].mean()
    plt.axhline(y=overall_mean, color='r', linestyle='--', label='Overall Mean')
    plt.title(f'Boxplot of {predictor} by Churn')
    plt.xlabel('Churn')
    plt.ylabel(predictor)
    plt.legend()
    plt.show()

# Define interval predictors
interval_predictors = ['MonthlyCharges', 'Tenure', 'TotalCharges']

# Generate boxplots for each interval predictor
for predictor in interval_predictors:
    plot_boxplots_by_target(data, predictor)

# Load the data for model training
trainData = pd.read_excel("Telco-Customer-Churn.xlsx")

# Define categorical and interval predictors
catName = ['Contract', 'Dependents', 'Gender', 'InternetService', 'MultipleLines', 'PaperlessBilling', 'Partner', 'PhoneService', 'SeniorCitizen']
intName = ['MonthlyCharges', 'Tenure', 'TotalCharges']
yName = 'Churn'

# Preprocess categorical predictors
for pred in catName:
    u = trainData[pred].astype('category').copy()
    u_freq = u.value_counts(ascending=True)
    trainData[pred] = u.cat.reorder_categories(list(u_freq.index)).copy()

# Prepare for model selection
nPredictor = len(catName) + len(intName)
maxIter = 50
tolS = 1e-7

allComb = []
for r in range(nPredictor + 1):
    allComb.extend(list(combinations(catName + intName, r)))

startTime = time.time()

allCombResult = []
nComb = len(allComb)
for r in range(nComb):
    modelTerm = list(allComb[r])
    modelData = trainData[[yName] + modelTerm].dropna()
    n_sample = modelData.shape[0]

    X_train = modelData[[]].copy()
    X_train.insert(0, 'Intercept', 1.0)

    y_train = modelData[yName].map({'No': 0, 'Yes': 1})

    for pred in modelTerm:
        if pred in catName:
            X_train = X_train.join(pd.get_dummies(modelData[[pred]].astype('category'), dtype=float))
        else:
            X_train = X_train.join(modelData[pred])

    resultList = Regression.BinaryLogisticRegression(X_train, y_train, maxIter=50)

    modelLLK = resultList[3]
    modelDF = len(resultList[4])

    AIC = 2.0 * modelDF - 2.0 * modelLLK
    BIC = modelDF * np.log(n_sample) - 2.0 * modelLLK

    allCombResult.append([modelTerm, len(modelTerm), modelLLK, modelDF, AIC, BIC])

endTime = time.time()

# Convert results to DataFrame
allCombResult_df = pd.DataFrame(allCombResult, columns=['Model Term', 'N Model Term', 'Log-Likelihood',
                                                        'Model Degree of Freedom', 'Akaike Information Criterion',
                                                        'Bayesian Information Criterion'])

# Sort models by ascending BIC values
allCombResult_df.sort_values(by=['Bayesian Information Criterion', 'N Model Term'], ascending=True, inplace=True)

# Save results to CSV file
allCombResult_df.to_csv('model_selection_summary.csv', index=False)

# Final model selection
chosen_model = allCombResult_df.iloc[0]['Model Term']
modelData = trainData[[yName] + chosen_model].dropna()
n_sample = modelData.shape[0]

X_train = modelData[[]].copy()
X_train.insert(0, 'Intercept', 1.0)

for pred in chosen_model:
    if pred in catName:
        X_train = X_train.join(pd.get_dummies(modelData[[pred]].astype('category'), dtype=float))
    else:
        X_train = X_train.join(modelData[pred])

y_train = modelData[yName].map({'No': 0, 'Yes': 1})
resultList = Regression.BinaryLogisticRegression(X_train, y_train, maxIter=50)

paramEstimate = resultList[0]
modelLLK = resultList[3]
modelDF = len(resultList[4])

elapsedTime = endTime - startTime

# Print chosen model
print("Chosen model based on BIC value:")
print(chosen_model)
print("Number of non-aliased parameters:", modelDF)
print("Log-Likelihood value:", modelLLK)
print("AIC value:", 2.0 * modelDF - 2.0 * modelLLK)
print("BIC value:", modelDF * np.log(n_sample) - 2.0 * modelLLK)

customer_profile = pd.Series({
    'Contract': 'Month-to-month', 'Dependents': 'No', 'Gender': 'Male',
    'InternetService': 'Fiber optic', 'MultipleLines': 'No phone service',
    'PaperlessBilling': 'Yes', 'Partner': 'No', 'PhoneService': 'No',
    'SeniorCitizen': 'Yes', 'MonthlyCharges': 70, 'Tenure': 29, 'TotalCharges': 1400
})

# Initialize X_test with zeros
X_test = pd.DataFrame(np.zeros((1, len(X_train.columns))), columns=X_train.columns)

# Set Intercept column to 1.0
X_test['Intercept'] = 1.0

# Populate X_test with values from customer_profile
for pred in customer_profile.index:
    if pred in modelTerm:
        pred_value = customer_profile[pred]
        if pred in catName:
            pname = pred + '_' + str(pred_value)
            if pname in X_test.columns:
                X_test[pname] = 1.0
        else:
            if pred in X_test.columns:
                X_test[pred] = pred_value

# Ensure that X_test has the same columns as X_train
X_test = X_test[X_train.columns]

# Check the number of features in X_test and beta
if X_test.shape[1] != len(paramEstimate):
    raise ValueError("Number of features in X_test does not match the number of coefficients in beta")

# Perform dot product
beta = paramEstimate['Estimate'].to_numpy()
nu = X_test.dot(beta)

# Calculate odds and probabilities
odds = np.exp(nu)
y_p0 = 1.0 / (1.0 + odds)
y_p1 = 1.0 - y_p0
print(y_p1)


# Question 3

predprob_event = resultList[6][1]

residual_simple = np.where(y_train == 1, 1.0, 0.0) - predprob_event
denom = np.sqrt(predprob_event * (1.0 - predprob_event))
residual_pearson = np.where(denom > 0.0, residual_simple / denom, np.nan)

r_p1 = np.where(predprob_event > 0.0, y_train * np.log(predprob_event), np.where(y_train == 0, 0.0, np.nan))
r_p2 = np.where(predprob_event < 1.0, (1 - y_train) * np.log(1.0 - predprob_event), np.where(y_train == 1, 0.0, np.nan))
sqrt_arg = -2.0 * (r_p1 + r_p2)
residual_deviance = np.where(np.logical_or(np.isnan(sqrt_arg), sqrt_arg < 0.0), np.nan, np.sign(residual_simple) * np.sqrt(sqrt_arg))

plot_data = data[[yName]].copy()
plot_data['Probability'] = predprob_event 
plot_data['Simple'] = residual_simple.copy()
plot_data['Pearson'] = residual_pearson
plot_data['Deviance'] = residual_deviance

# Plotting boxplots for each type of residual
sns.set(style="whitegrid")
boxplot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
fig, ax = plt.subplots(figsize=(10, 4))
plot_data.boxplot(column='Probability', by=yName, vert=False, ax=ax, color=boxplot_colors[0])
ax.set_xlabel('Predicted Probability of Churn')
ax.set_ylabel('Churn')
ax.set_xticks(np.arange(0.0, 1.1, 0.1))
plt.suptitle('')
plt.title('')
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
plot_data.boxplot(column='Simple', by=yName, vert=False, ax=ax, color=boxplot_colors[1])
ax.set_xlabel('Simple Residual')
ax.set_ylabel('Churn')
plt.suptitle('')
plt.title('')
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
plot_data.boxplot(column='Pearson', by=yName, vert=False, ax=ax, color=boxplot_colors[2])
ax.set_xlabel('Pearson Residual')
ax.set_ylabel('Churn')
plt.suptitle('')
plt.title('')
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
plot_data.boxplot(column='Deviance', by=yName, vert=False, ax=ax, color=boxplot_colors[3])
ax.set_xlabel('Deviance Residual')
ax.set_ylabel('Churn')
plt.suptitle('')
plt.title('')
plt.show()


# Additional Questions

nullLLK = allCombResult_df[allCombResult_df['N Model Term'] == 0]['Log-Likelihood'][0]
R_MF = 1.0 - (modelLLK / nullLLK)
exparg = (2.0 / n_sample) * (nullLLK - modelLLK)
R_CS = np.exp(exparg)
R_CS = 1.0 - R_CS
upbound = 1.0 - np.exp(nullLLK * 2.0 / n_sample)
R_N = R_CS / upbound
S1 = np.mean(predprob_event[y_train == 1])
S0 = np.mean(predprob_event[y_train == 0])
R_TJ = S1 - S0

# Results
results = [
    ["McFadden R-squared", R_MF],
    ["Cox-Snell R-squared", R_CS],
    ["Nagelkerke R-squared", R_N],
    ["Tjur Coefficient of Discrimination", R_TJ]
]

# Print the table
print(tabulate(results, headers=["Metric", "Value"], tablefmt="fancy_grid"))


# AUC Calculation
auc_value = roc_auc_score(y_train, predprob_event)
print("Area Under Curve (AUC) value:", auc_value)
if auc_value >= 0.5:
    print("The final model is acceptable based on the AUC value.")
else:
    print("The final model is not acceptable based on the AUC value.")


# RASE Calculation
squared_errors = (y_train - predprob_event) ** 2
mean_squared_error = np.mean(squared_errors)
rase_value = np.sqrt(mean_squared_error)
print("Root Average Squared Error (RASE) value:", rase_value)
if rase_value <= 0.5:
    print("The final model is acceptable based on the RASE value.")
else:
    print("The final model is not acceptable based on the RASE value.")


# ROC Curve
outCurve = Regression.curve_coordinates(y_train, 1, 0, predprob_event)
fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
ax.plot(outCurve['OneMinusSpecificity'], outCurve['Sensitivity'], marker='+', markersize=2, color='blue')
ax.axline([0, 0], slope=1, color='red', linestyle='--')
ax.set_xlabel('One Minus Specificity (False Positive Rate)')
ax.set_ylabel('Sensitivity (True Positive Rate)')
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
ax.set_xticks(np.arange(0.0, 1.1, 0.1))
ax.yaxis.grid(True, which='major')
ax.xaxis.grid(True, which='major')
plt.show()


# Kolmogorov-Smirnov statistic
KS_stat = outCurve['Sensitivity'] - outCurve['OneMinusSpecificity']
ipos = KS_stat.argmax()
row = outCurve.iloc[ipos]
KS_threshold = row['Threshold']
outMetrics = Regression.binary_model_metric(y_train, 1, 0, predprob_event, KS_threshold)
print('Kolmogorov-Smirnov statistic = ', KS_stat[ipos])
print('Kolmogorov-Smirnov Threshold = ', KS_threshold)
print('Area Under Curve = ', outMetrics['AUC'])
print('Root Average Squared Error = ', outMetrics['RASE'])
print('Misclassification Rate = ', outMetrics['MCE'])


# Precision-Recall Curve
yName = 'Churn'
yCat = ['No', 'Yes']

inputData = pandas.read_excel('Telco-Customer-Churn.xlsx')

trainData = inputData[catName + intName + [yName]].dropna().reset_index(drop = True)
n_sample = trainData.shape[0]

yFreq = trainData.groupby(yName).size()
overall_odds = yFreq['Yes'] / yFreq['No']

noskill = yFreq['Yes'] / (yFreq['Yes'] + yFreq['No'])
fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
ax.plot(outCurve['Recall'], outCurve['Precision'], marker='+', markersize=2, color='blue')
ax.axline([0, noskill], slope=0, color='red', linestyle='--', label='No Skill')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
ax.set_xticks(np.arange(0.0, 1.1, 0.1))
ax.yaxis.grid(True, which='major')
ax.xaxis.grid(True, which='major')
plt.legend()
plt.show()

idx = outCurve['F1 Score'].idxmax()
F1_maxscore = outCurve.iloc[idx]['F1 Score']
F1_threshold = outCurve.iloc[idx]['Threshold']
print('F1 Max Score = ', F1_maxscore)

outMetrics = Regression.binary_model_metric(y_train, 1, 0, predprob_event, F1_threshold)
print('F1 Threshold = ', F1_threshold)
print('Area Under Curve = ', outMetrics['AUC'])
print('Root Average Squared Error = ', outMetrics['RASE'])
print('Misclassification Rate = ', outMetrics['MCE'])

outMetrics = Regression.binary_model_metric(y_train, 1, 0, predprob_event, 0.5)
print('Default Threshold = ', 0.5)
print('Area Under Curve = ', outMetrics['AUC'])
print('Root Average Squared Error = ', outMetrics['RASE'])
print('Misclassification Rate = ', outMetrics['MCE'])

outMetrics = Regression.binary_model_metric(y_train, 1, 0, predprob_event, noskill)
print('Noskill Threshold = ', noskill)
print('Area Under Curve = ', outMetrics['AUC'])
print('Root Average Squared Error = ', outMetrics['RASE'])
print('Misclassification Rate = ', outMetrics['MCE'])

