"""
===============================================================================
Home Credit Scoring Model Project: Modelling
===============================================================================

This file is organised as follows:
1. Load the dataset
2. Feature Engineering
3. Machine Learning
   3.1 Cost function
   3.2 PyCaret: Comparison of classifiers by cross-validation
   3.3 FLAML: Selection of the final model
       3.3.1 Optimisation of models using the ROC AUC score
       3.3.2 Optimisation of models using the cost function
"""
# Standard libraries
import random
import time
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import pycaret
import flaml
import shap
import joblib
import pickle


from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import TargetEncoder, RobustScaler
from sklearn.metrics import fbeta_score, precision_recall_curve
from collinearity import SelectNonCollinear
from pycaret.classification import *
from flaml import AutoML
from shap import Explainer, maskers, plots
from joblib import dump
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('PyCaret: {}'.format(pycaret.__version__))
print('FLAML: {}'.format(flaml.__version__))
print('SHAP: {}'.format(shap.__version__))
print('Joblib: {}'.format(joblib.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150
FOLDS = 10
BETA = 2.0

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Set the maximum number of rows and columns to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Load the dataset
===============================================================================
"""
print(f'\n\n\n1. Load the dataset')

# Load the dataset
INPUT_CSV = 'datasets/customers_dataset.csv'
customers_dataset = load_dataset(INPUT_CSV, 'utf-8')



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
print(f'\n\n\n2. Feature Engineering')

# Feature selection
y = customers_dataset['TARGET'].values
X = customers_dataset.drop(['TARGET', 'SK_ID_CURR', 'CODE_GENDER'], axis=1)
X.columns = X.columns.str.replace('_', ' ')

# Display X dataset information and description
dataset_info_description(X, 15)

numeric_features = X.select_dtypes(include=['number']).columns.tolist()
object_features = X.select_dtypes(include=['object']).columns.tolist()
print('\n\nLength of numeric features: {}'.format(len(numeric_features)))
print('Numeric features:\n{}'.format(numeric_features))
print('\nLength of object features: {}'.format(len(object_features)))
print('Object features:\n{}'.format(object_features))


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True, stratify=y)


# Encode object features
encoder = TargetEncoder(cv=FOLDS, random_state=SEED)
object_train_encoded = encoder.fit_transform(
    X=X_train[object_features], y=y_train)
object_test_encoded = encoder.transform(X=X_test[object_features])

# Encoder persistence
dump(encoder, 'models/encoder/encoder.joblib')


# Add encoded features to the numerical dataset
X_train_numeric = X_train[numeric_features]
X_test_numeric = X_test[numeric_features]
for i, feature in enumerate(object_features):
    X_train_numeric[feature] = object_train_encoded[:, i]
    X_test_numeric[feature] = object_test_encoded[:, i]


# Normalisation
scaler = RobustScaler()
train_scaled = scaler.fit_transform(X_train_numeric)
test_scaled = scaler.transform(X_test_numeric)

# Scaler persistence
dump(scaler, 'models/scaler/scaler.joblib')


# Convert arrays into DataFrames
train_dataset = pd.DataFrame(
    data=train_scaled, columns=list(X_train_numeric.columns))
test_dataset = pd.DataFrame(
    data=test_scaled, columns=list(X_test_numeric.columns))


# Get a set of correlated features based on the correlation threshold
relevant_features = [
    'AMT CREDIT', 'AMT ANNUITY', 'AMT INCOME TOTAL', 'DAYS EMPLOYED',
    'ORGANIZATION TYPE', 'DAYS BIRTH', 'NAME FAMILY STATUS', 'CNT CHILDREN'
]
X_train_selected = np.array(train_dataset.drop(relevant_features, axis=1))
X_test_selected = np.array(test_dataset.drop(relevant_features, axis=1))
selector = SelectNonCollinear(correlation_threshold=0.9, scoring=f_classif)
selector.fit(X_train_selected, y_train)
train_dataset_selected = selector.transform(X_train_selected)
test_dataset_selected = selector.transform(X_test_selected)
selected_features = list(np.array(train_dataset.drop(
    relevant_features, axis=1).columns)[selector.get_support()])
features = list()
features.extend(relevant_features)
features.extend(selected_features)
correlated_features = list(set(train_dataset.columns) - set(selected_features))
train_dataset = train_dataset[features]
test_dataset = test_dataset[features]
print('\n\nLength of Correlated features: {}'.format(len(correlated_features)))
print('Correlated features:\n{}'.format(correlated_features))

# Scaler persistence
dump(selector, 'models/selector/selector.joblib')


# Display train dataset information and description
dataset_info_description(train_dataset, 15)

# Display test dataset information and description
dataset_info_description(test_dataset, 15)


# Add labels to the training and test datasets
train_dataset = train_dataset.assign(LABEL=y_train)
test_dataset = test_dataset.assign(LABEL=y_test)

# Display train dataset information and description
dataset_info_description(train_dataset, 15)

# Display test dataset information and description
dataset_info_description(test_dataset, 15)

# Save the training and test datasets in CSV format
train_dataset.to_csv('datasets/train_dataset.csv', index=False)
test_dataset.to_csv('datasets/test_dataset.csv', index=False)



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
print(f'\n\n\n3. Machine Learning')

# Classes and labels
print(f'\n\nTrain classes count: {Counter(y_train)}')
print(f'Test classes count: {Counter(y_test)}')
labels = list(set(y_test))
print(f'Labels: {labels}')


# 3.1 Cost function
print(f'\n\n3.1 Cost function')
"""
The cost function is an estimation of the financial gain or loss for 
the company. This is a business problem and the purpose is to minimise:
- The number of high-risk customers whose loans are granted in error 
  (type II error or false negatives).
- The number of risk-free customers whose credit applications are rejected in 
  error (type I error or false positives).

To achieve this objective, it is necessary to:
- Maximise recall while maintaining acceptable precision.
- Minimise the false positives rate (FPR) and the false negatives rate (FNR).

The evaluation metric is a loss function calculated from the F-beta score: 
Fβ = (1 + β**2) * precisions * recalls / (β**2 * precisions + recalls)

This formula is derived from the following formula:
Fβ = (1 + β**2) * tp / ((1 + β**2) * tp + fp + β**2 * fn)
Where:
- TP (True Positives): the number of samples that are actually positive and 
                       are predicted as positive.
- FP (False Positives): the number of samples that are actually negative but 
                        are predicted as positive.
- FN (False Negatives): the number of samples that are actually positive but 
                        are predicted as negative. 

The challenge is to find the right compromise between the recall and precision. 
This depends on the value of the parameter β:
- If β < 1, precision has more weight in the evaluation (the false positives 
  are more penalised).
- If β = 1, precision and recall have the same weight in the evaluation (in 
  this case, the result is F1-score).
- If β > 1, recall has more weight in the evaluation (the false negatives 
  are more penalised).
"""

def fbeta(y, y_pred, **kwargs):
    """This function computes the F-beta score.

    Args:
        y (array-like): the true labels
        y_pred (array-like): the predicted labels
        **kwargs: additional arguments
            - BETA (float): the weighted harmonic mean of precision and recall
                            defines the influence of precision or recall on the
                            evaluation. Its determines the weight of precision
                            or recall in the combined score
            - labels (array-like): the list of labels for which to compute the
                                   score

    Returns:
        float: the computed F-beta score
    """
    return fbeta_score(y_true=y, y_pred=y_pred, beta=BETA, labels=labels)


def cost_function(X_val, y_val, estimator, labels, X_train, y_train, *args):
    """This function computes a custom evaluation metric combining validation
    and training log loss, and evaluates a given estimator by computing a loss
    on both validation and training sets, then returns a weighted combination
    of these losses. It also measures the prediction time.

    Args:
        X_val (array-like or pd.DataFrame): the validation feature matrix
        y_val (array-like): the validation labels
        estimator: the fitted estimator
        labels (array-like): the class labels
        X_train (array-like or pd.DataFrame): the training feature matrix
        y_train (array-like): the training labels
        *args (variable length argument list): additional arguments
            - BETA (float): the weighted harmonic mean of precision and recall
                            defines the influence of precision or recall on the
                            evaluation. Its determines the weight of precision
                            or recall in the combined score

    Returns:
        tuple (custom_metric, metrics_dict):
            - custom_metric (float): the weighted combination of validation
                                     and training log loss
            - metrics_dict (dict): the dictionary containing:
                - val_loss (float): the loss on validation set
                - train_loss (float): the loss on training set
                - pred_time (float): the prediction time (seconds)
    """

    start = time.time()
    y_proba = estimator.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    fbeta_scores = (
        (1 + BETA**2) * precisions * recalls / (BETA**2 * precisions + recalls)
    )
    best_threshold = thresholds[np.nanargmax(fbeta_scores)]
    y_pred = np.where(y_proba > best_threshold, 1, 0)
    pred_time = (time.time() - start) / len(X_val)
    val_loss = 1 - fbeta_score(y_val, y_pred, beta=BETA, labels=labels)

    y_proba = estimator.predict_proba(X_train)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_proba)
    fbeta_scores = (
        (1 + BETA**2) * precisions * recalls / (BETA**2 * precisions + recalls)
    )
    best_threshold = thresholds[np.nanargmax(fbeta_scores)]
    y_pred = np.where(y_proba > best_threshold, 1, 0)
    train_loss = 1 - fbeta_score(y_train, y_pred, beta=BETA, labels=labels)
    return val_loss, {
        'val_loss': val_loss, 'train_loss': train_loss, 'pred_time': pred_time}


# 3.2 PyCaret: Comparison of classifiers by cross-validation
print(f'\n\n3.2 PyCaret: Comparison of classifiers by cross-validation')

# Set up the setup
s = setup(
    data=train_dataset,
    target='LABEL',
    index=False,
    train_size=0.8,
    preprocess=False,
    fold=FOLDS,
    fold_shuffle=True,
    session_id=SEED,
    verbose=True
)

add_metric(
    id='F-beta score',
    name='F-BETA',
    score_func=fbeta,
    target='pred',
    greater_is_better=True,
    multiclass=False
)

# Selection of the best model by cross-validation
best = compare_models(
    fold=FOLDS,
    round=3,
    cross_validation=True,
    n_select=1,
    sort='AUC',
    verbose=True
)
print(f'\nClassification of models:\n{best}')

# Make predictions
pred = predict_model(estimator=best, data=test_dataset)
print(f'\nPredictions:\n{pred}')

# Plot confusion matrix
try:
    plot_model(best, plot='confusion_matrix')
except Exception as error:
    print(f'The following error occurred: {error}')

# Plot class report
try:
    plot_model(best, plot='class_report')
except Exception as error:
    print(f'The following error occurred: {error}')

# Plot AUC
try:
    plot_model(best, plot='auc')
except Exception as error:
    print(f'The following error occurred: {error}')

# Plot the global interpretability of the model (Feature importance)
try:
    plot_model(estimator=best, plot='feature')
except Exception as error:
    print(f'The following error occurred: {error}')

# Plot the global interpretability of the model (Summary)
try:
    interpret_model(estimator=best, plot='summary')
except Exception as error:
    print(f'The following error occurred: {error}')

# Make predictions
y_pred = pred['prediction_label'].to_numpy()
if hasattr(best, 'predict_proba'):
    y_proba = best.predict_proba(test_dataset.drop(['LABEL'], axis=1))[:, 1]
else:
    y_proba=None

# Evaluation
evaluate_binary_classification(y_test, y_pred, y_proba, BETA, labels)

# Dashboard
#dashboard(best, display_format='inline')

# Create Gradio App
#create_app(best)

# Model persistence: save the pipeline
save_model(best, 'models/pycaret/model')


# 3.3 FLAML: Selection of the final model
print(f'\n\n3.3 FLAML: Selection of the final model')


# 3.3.1 Optimisation of models using the ROC AUC score
print(f'\n3.3.1 Optimisation of models using the ROC AUC score')

# Instantiate AutoML instance
automl = AutoML()
automl.fit(
    dataframe=train_dataset,
    label='LABEL',
    metric='roc_auc',
    task='classification',
    n_jobs=-1,
    eval_method='auto',
    n_splits=FOLDS,
    split_type='auto',
    seed=SEED,
    early_stop=True
)

# Display information about the best model
print('\nBest estimator: {}'.format(automl.best_estimator))
print('Best hyperparameters:\n{}'.format(automl.best_config))
print('Best loss: {}'.format(automl.best_loss))
print('Training time: {}s'.format(automl.best_config_train_time))

# Plot the global interpretability of the model (Feature importance)
try:
    feature_importance_viz = pd.DataFrame(
        data={'Importance': automl.model.estimator.feature_importances_},
        index=automl.model.estimator.feature_names_in_
    )
    feature_importance_viz = feature_importance_viz.sort_values(
        by=['Importance'], ascending=True)
    feature_importance_viz = feature_importance_viz[-50:]
    ax = feature_importance_viz.plot.barh()
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.legend(loc='best')
    ax.grid(True)
    plt.show()
except Exception as error:
    print(f'The following error occurred: {error}')

# Make predictions
y_pred = automl.predict(test_dataset.drop(['LABEL'], axis=1))
if hasattr(automl, 'predict_proba'):
    y_proba = automl.predict_proba(test_dataset.drop(['LABEL'], axis=1))[:, 1]
else:
    y_proba=None

# Evaluation
evaluate_binary_classification(y_test, y_pred, y_proba, BETA, labels)

# Plot the ROC curve
display_roc_curve(y_test, y_proba)


# 3.3.2 Optimisation of models using the cost function
print(f'\n3.3.2 Optimisation of models using the cost function')

# Instantiate AutoML instance
automl = AutoML()
automl.fit(
    dataframe=train_dataset,
    label='LABEL',
    metric=cost_function,
    task='classification',
    n_jobs=-1,
    estimator_list=['lrl1', 'lrl2', 'sgd', 'lgbm', 'xgboost', 'catboost'],
    eval_method='auto',
    n_splits=FOLDS,
    split_type='auto',
    seed=SEED,
    early_stop=True
)

# Display information about the best model
print('\nBest estimator: {}'.format(automl.best_estimator))
print('Best hyperparameters:\n{}'.format(automl.best_config))
print('Best loss: {}'.format(automl.best_loss))
print('Training time: {}s'.format(automl.best_config_train_time))

# Make predictions
y_pred = automl.predict(test_dataset.drop(['LABEL'], axis=1))
if hasattr(automl, 'predict_proba'):
    y_proba = automl.predict_proba(test_dataset.drop(['LABEL'], axis=1))[:, 1]
else:
    y_proba=None

# Evaluation
evaluate_binary_classification(y_test, y_pred, y_proba, BETA, labels)

# Calculate the best risk threshold that maximises fbeta score
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
fbeta_scores = (
    (1 + BETA**2) * precisions * recalls / (BETA**2 * precisions + recalls)
)
best_threshold = thresholds[np.nanargmax(fbeta_scores)]
credit_score_best_risk_threshold = int(best_threshold * 100)
print(f'\nBest risk threshold of the credit score: '
      f'{credit_score_best_risk_threshold}')

# Evaluation after calculation of the best risk threshold
y_pred = np.where(y_proba > best_threshold, 1, 0)
evaluate_binary_classification(y_test, y_pred, y_proba, BETA, labels)

# Plot the ROC curve
display_roc_curve(y_test, y_proba)

# Plot the global interpretability of the model
try:
    explainer = Explainer(
        model=automl.model.estimator,
        masker=maskers.Independent(
            data=train_dataset.drop(['LABEL'], axis=1), max_samples=1000)
    )
    shap_values = explainer(test_dataset.drop(['LABEL'], axis=1))
    plots.beeswarm(shap_values=shap_values, max_display=50)
except Exception as error:
    print(f'The following error occurred: {error}')

# Plot the local interpretability of the model
try:
    explainer = Explainer(
        model=automl.model.estimator,
        masker=maskers.Independent(
            data=train_dataset.drop(['LABEL'], axis=1), max_samples=1000)
    )
    shap_values = explainer(test_dataset.drop(['LABEL'], axis=1))
    plots.waterfall(shap_values=shap_values[0], max_display=50)
except Exception as error:
    print(f'The following error occurred: {error}')

# Model persistence
automl_path = 'models/flaml/automl.pkl'
with open(automl_path, 'wb') as f:
    pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
dump(automl.model.estimator, 'models/flaml/model.joblib')

# Save the best risk threshold of the credit score into a NumPy file
np.save(
    file='models/flaml/results/credit_score_best_risk_threshold.npy',
    arr=np.array(credit_score_best_risk_threshold)
)
