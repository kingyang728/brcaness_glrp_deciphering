# Import required libraries for performance metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
# Import required libraries for machine learning classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Import XGBoost and CatBoost classifiers
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Define dictionary with performance metrics
scoring = {'accuracy':make_scorer(accuracy_score),
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score),
           'f1_score':make_scorer(f1_score)}



# Instantiate the machine learning classifiers
log_model = LogisticRegression(max_iter=10000)
svc_model = LinearSVC(dual=False)
dtr_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
# rfc_model = RandomForestClassifier(n_estimators= 60, max_depth=5, min_samples_split=50,
#                                  min_samples_leaf=20 ,oob_score=True, max_features=3,random_state=10)
# rfc_model = RandomForestClassifier(n_estimators= 187, max_depth=4, min_samples_split=130,
#                             min_samples_leaf=11 ,max_features = 4, n_jobs=-1 ,
#                             oob_score=True, criterion = 'gini',random_state=10)
gnb_model = GaussianNB()


## XGboost and Catboost 
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
cat_model = CatBoostClassifier(silent=True)

# def run_model(model, X_train, y_train, X_test, y_test):
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     scores = {
#         'accuracy': accuracy_score(y_test, predictions),
#         'precision': precision_score(y_test, predictions, average='binary'),
#         'recall': recall_score(y_test, predictions, average='binary'),
#         'f1_score': f1_score(y_test, predictions, average='binary')
#     }
#     return scores

# Define the models evaluation function
def models_evaluation(X_train, y_train, X_test, y_test,feature_names):
    '''
    X_train : training data set features
    y_train : training data set target
    X_test : testing data set features
    y_test : testing data set target
    feature_names : names of the features
    '''
    
    rfc_model.fit(X_train, y_train)
    log_model.fit(X_train, y_train)
    svc_model.fit(X_train, y_train)
    dtr_model.fit(X_train, y_train)
    gnb_model.fit(X_train, y_train)

    # Predict on test data
    rfc_pred = rfc_model.predict(X_test)
    log_pred = log_model.predict(X_test)
    svc_pred = svc_model.predict(X_test)
    dtr_pred = dtr_model.predict(X_test)
    gnb_pred = gnb_model.predict(X_test)

    # Feature Importance for Random Forest and Decision Tree
    rfc_importance = rfc_model.feature_importances_
    dtr_importance = dtr_model.feature_importances_

    # For Logistic Regression and Linear SVC, we can use the coefficients as an indicator of feature importance
    log_importance = np.abs(log_model.coef_[0])
    svc_importance = np.abs(svc_model.coef_[0])

    # Compute scores based on predictions
    models_scores_table = pd.DataFrame({
        'Logistic Regression': [accuracy_score(y_test, log_pred), precision_score(y_test, log_pred), recall_score(y_test, log_pred), f1_score(y_test, log_pred)],
        'Support Vector Classifier': [accuracy_score(y_test, svc_pred), precision_score(y_test, svc_pred), recall_score(y_test, svc_pred), f1_score(y_test, svc_pred)],
        'Decision Tree': [accuracy_score(y_test, dtr_pred), precision_score(y_test, dtr_pred), recall_score(y_test, dtr_pred), f1_score(y_test, dtr_pred)],
        'Random Forest': [accuracy_score(y_test, rfc_pred), precision_score(y_test, rfc_pred), recall_score(y_test, rfc_pred), f1_score(y_test, rfc_pred)],
        'Gaussian Naive Bayes': [accuracy_score(y_test, gnb_pred), precision_score(y_test, gnb_pred), recall_score(y_test, gnb_pred), f1_score(y_test, gnb_pred)]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])


    # Add 'Best Model' column
    # models_scores_table['Best Model'] = models_scores_table.idxmax(axis=1)

    # Creating a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Gene': feature_names,
        'RandomForest': rfc_importance,
        'DecisionTree': dtr_importance,
        'LogisticRegression': log_importance,
        'SVC': svc_importance
    })

    # Return both models performance metrics scores and feature importance data frame
    return models_scores_table,feature_importance_df


# Define the models evaluation function
def models_evaluation_withXGB_Cat(X_train, y_train, X_test, y_test,feature_names):
    '''
    X_train : training data set features
    y_train : training data set target
    X_test : testing data set features
    y_test : testing data set target
    feature_names : names of the features
    '''
    
    rfc_model.fit(X_train, y_train)
    log_model.fit(X_train, y_train)
    svc_model.fit(X_train, y_train)
    dtr_model.fit(X_train, y_train)
    gnb_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    cat_model.fit(X_train, y_train)

    # Predict on test data
    rfc_pred = rfc_model.predict(X_test)
    log_pred = log_model.predict(X_test)
    svc_pred = svc_model.predict(X_test)
    dtr_pred = dtr_model.predict(X_test)
    gnb_pred = gnb_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    cat_pred = cat_model.predict(X_test)


    # Feature Importance for Random Forest and Decision Tree
    rfc_importance = rfc_model.feature_importances_
    dtr_importance = dtr_model.feature_importances_
    xgb_importance = xgb_model.feature_importances_
    cat_importance = cat_model.feature_importances_

    # For Logistic Regression and Linear SVC, we can use the coefficients as an indicator of feature importance
    log_importance = np.abs(log_model.coef_[0])
    svc_importance = np.abs(svc_model.coef_[0])

    # Compute scores based on predictions
    models_scores_table = pd.DataFrame({
        'Logistic Regression': [accuracy_score(y_test, log_pred), precision_score(y_test, log_pred), recall_score(y_test, log_pred), f1_score(y_test, log_pred)],
        'Support Vector Classifier': [accuracy_score(y_test, svc_pred), precision_score(y_test, svc_pred), recall_score(y_test, svc_pred), f1_score(y_test, svc_pred)],
        'Decision Tree': [accuracy_score(y_test, dtr_pred), precision_score(y_test, dtr_pred), recall_score(y_test, dtr_pred), f1_score(y_test, dtr_pred)],
        'Random Forest': [accuracy_score(y_test, rfc_pred), precision_score(y_test, rfc_pred), recall_score(y_test, rfc_pred), f1_score(y_test, rfc_pred)],
        'Gaussian Naive Bayes': [accuracy_score(y_test, gnb_pred), precision_score(y_test, gnb_pred), recall_score(y_test, gnb_pred), f1_score(y_test, gnb_pred)],
        'XGBoost': [accuracy_score(y_test, xgb_pred), precision_score(y_test, xgb_pred), recall_score(y_test, xgb_pred), f1_score(y_test, xgb_pred)],
        'CatBoost': [accuracy_score(y_test, cat_pred), precision_score(y_test, cat_pred), recall_score(y_test, cat_pred), f1_score(y_test, cat_pred)]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])


    # Add 'Best Model' column
    # models_scores_table['Best Model'] = models_scores_table.idxmax(axis=1)

    # Creating a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Gene': feature_names,
        'RandomForest': rfc_importance,
        'DecisionTree': dtr_importance,
        'LogisticRegression': log_importance,
        'SVC': svc_importance,
        'XGBoost': xgb_importance,
        'CatBoost': cat_importance
    })

    # Return both models performance metrics scores and feature importance data frame
    return models_scores_table,feature_importance_df