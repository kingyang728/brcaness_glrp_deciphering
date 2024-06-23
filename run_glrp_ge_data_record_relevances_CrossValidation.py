#!python

"""
Running the GLRP on GCNN model trained on gene expression data. 90% is for training and 10% for testing. 
Relevances obtained for 10% of testing patients are written into the file "relevances_rendered_class.csv". From these relevances the patient subnetworks can be built.
The file "predicted_concordance.csv" contains a table showing which patients were predicted correctly.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ML_Evaluation import models_evaluation

from components import data_handling, glrp_scipy
from lib import models, graph, coarsening

# from sklearn.model_selection import StratifiedKFold

import time

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from lib import models, graph, coarsening

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from components import data_handling, glrp_scipy
from scipy.sparse import csr_matrix

# Define dictionary with performance metrics
scoring = {'accuracy':make_scorer(accuracy_score),
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score),
           'f1_score':make_scorer(f1_score)}

# Import required libraries for machine learning classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

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

### Function to coarsen the graph and permute data, it should be called before training the models
# def prepare_graph_data(X, A):
#     graphs, perm = coarsening.coarsen(A, levels=2, self_connections=False)
#     L = [graph.laplacian(A, normalized=True) for A in graphs]
#     X_perm = coarsening.perm_data(X, perm)
#     return X_perm, L

# Define the models evaluation function
def models_evaluation_pre(X_train, y_train, X_test, y_test,feature_names,A):
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

    # GCNN training and prediction

    # coarsening the PPI graph to perform pooling in the model
    graphs, perm = coarsening.coarsen(A, levels=2, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]

    X_train_GCNN = coarsening.perm_data(X_train, perm)
    X_test_GCNN = coarsening.perm_data(X_test, perm)

    n_train = X_train_GCNN.shape[0]
        # Building blocks.
    params = dict()
    params['dir_name']       = 'TCGA_BRCAness'
    params['num_epochs']     = 100
    # params['batch_size']     = 100
    params['batch_size']     = 109
    # params['batch_size']     = 109
    params['eval_frequency'] = 40

    # Building blocks.
    params['filter']         = 'chebyshev5'
    params['brelu']          = 'b1relu'
    params['pool']           = 'mpool1'

    # Number of classes.
    C = y.max() + 1
    assert C == np.unique(y).size

    # Architecture.
    params['F']              = [32, 32]  # Number of graph convolutional filters.
    params['K']              = [8, 8]  # Polynomial orders.
    params['p']              = [2, 2]    # Pooling sizes.
    params['M']              = [512, 128, C]  # Output dimensionality of fully connected layers.

    # Optimization.
    params['regularization'] = 1e-4
    params['dropout']        = 1
    params['learning_rate']  = 1e-3
    params['decay_rate']     = 0.95
    params['momentum']       = 0
    params['decay_steps']    = n_train / params['batch_size']
    model = models.cgcnn(L, **params)
    

    start = time.time()
    accuracy, loss, t_step, trained_losses = model.fit(X_train_GCNN, y_train, X_test_GCNN, y_test)
    end = time.time()
    probas_ = model.get_probabilities(X_test_GCNN)
    labels_by_network = np.argmax(probas_, axis=1)
    gcnn_metrics = [
        accuracy_score(y_test, labels_by_network),
        precision_score(y_test, labels_by_network),
        recall_score(y_test, labels_by_network),
        f1_score(y_test, labels_by_network)
    ]


    # Compute scores based on predictions
    results = {
        'Logistic Regression': [accuracy_score(y_test, log_pred), precision_score(y_test, log_pred), recall_score(y_test, log_pred), f1_score(y_test, log_pred)],
        'Support Vector Classifier': [accuracy_score(y_test, svc_pred), precision_score(y_test, svc_pred), recall_score(y_test, svc_pred), f1_score(y_test, svc_pred)],
        'Decision Tree': [accuracy_score(y_test, dtr_pred), precision_score(y_test, dtr_pred), recall_score(y_test, dtr_pred), f1_score(y_test, dtr_pred)],
        'Random Forest': [accuracy_score(y_test, rfc_pred), precision_score(y_test, rfc_pred), recall_score(y_test, rfc_pred), f1_score(y_test, rfc_pred)],
        'Gaussian Naive Bayes': [accuracy_score(y_test, gnb_pred), precision_score(y_test, gnb_pred), recall_score(y_test, gnb_pred), f1_score(y_test, gnb_pred)]
    }
    results['GCNN'] = gcnn_metrics
    return results

rndm_state = 7
np.random.seed(rndm_state)
if __name__ == "__main__":

    # path_to_feature_val = "./Data_EXP_LRPData/TCGA_ALL_expected_countWithoutExpCluster/TCGA_exp_EXP.csv"
    # path_to_feature_graph = "./Data_EXP_LRPData/TCGA_ALL_expected_countWithoutExpCluster/TCGA_reactome_FIs.csv"
    # path_to_labels = "./Data_EXP_LRPData/TCGA_ALL_expected_countWithoutExpCluster/TCGA_BRCAness_Label.csv"

    path_to_feature_val = "./Data_EXP_LRPData/TCGA_BRCA/TCGA_exp_EXP.csv"
    path_to_feature_graph = "./Data_EXP_LRPData/TCGA_BRCA/TCGA_reactome_FIs.csv"
    path_to_labels = "./Data_EXP_LRPData/TCGA_BRCA/TCGA_BRCAness_Label.csv"
    
    path_to_comblabels = "./Data_EXP_LRPData/TCGA_BRCA/CancerEntity_combLabel.csv"

### set exp Brcaness labeled data paths here
#     path_to_feature_val = "~/harddisk/ICGC_Data/BRCA/BRCA-US/BRCA-US_EXP_LRPData/BRCA-US_exp_PPI.csv"
#     path_to_feature_graph = "~/harddisk/ICGC_Data/BRCA/BRCA-US/BRCA-US_EXP_LRPData/BRCA-US_HPRD_PPI.csv"
#     path_to_labels = "~/harddisk/ICGC_Data/BRCA/BRCA-US/BRCA-US_EXP_LRPData/BRCA-US_BRCAness_Label.csv"


    DP = data_handling.DataPreprocessor(path_to_feature_values=path_to_feature_val, path_to_feature_graph=path_to_feature_graph,
                                        path_to_labels=path_to_labels)
    X = DP.get_feature_values_as_np_array()  # gene expression
    A = csr_matrix(DP.get_adj_feature_graph_as_np_array().astype(np.float32))  # adjacency matrix of the PPI network
    y = DP.get_labels_as_np_array()  # labels

    print("GE data, X shape: ", X.shape)
    print("Labels, y shape: ", y.shape)
    print("PPI network adjacency matrix, A shape: ", A.shape)

#     X = X - np.min(X)
#     print(X)


    stratify_label = pd.read_csv(path_to_comblabels)
    # type(stratify_label)
    CombLabel_array = stratify_label.iloc[0].values
    len(CombLabel_array)
def Data_spilt(X, y, CombLabel_array):
  X_train_unnorm, X_test_unnorm, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                                      stratify=CombLabel_array, random_state=rndm_state)

  # Need to know which patients got into train and test subsets
  _, _, patient_indexes_train, patient_indexes_test = train_test_split(X, DP.labels.columns.values.tolist(), test_size=0.20,
                                                                      stratify=CombLabel_array, random_state=rndm_state)

  # Data frame with test patients and corresponding ground truth labels
  patient_ind_test_df = pd.DataFrame(data={"Patient ID": patient_indexes_test, "label": y_test})

  # !!!
  # Making data lying in the interval [0, 8.35]
  X_train = X_train_unnorm - np.min(X)
  X_test = X_test_unnorm - np.min(X)
  return X_train,X_test,y_train,y_test

def run_multiple_evaluations(X, y, feature_names, n_runs):
    metrics_data = {
        'Logistic Regression': [],
        'Support Vector Classifier': [],
        'Decision Tree': [],
        'Random Forest': [],
        'Gaussian Naive Bayes': [],
        'GCNN': [] 
    }

    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = Data_spilt(X, y, CombLabel_array)
        results = models_evaluation_pre(X_train, y_train, X_test, y_test, feature_names,A)
        for key, values in results.items():
            metrics_data[key].append(values)

    # Format the results to mean±std and find best model for each metric
    formatted_results = {}
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    for model, results in metrics_data.items():
        mean_results = np.mean(results, axis=0)
        std_results = np.std(results, axis=0)
        formatted_results[model] = [f"{mean:.4f}±{std:.4f}" for mean, std in zip(mean_results, std_results)]

    # Convert to DataFrame
    results_df = pd.DataFrame(formatted_results, index=metrics)

    # Calculate 'Best Model' based on mean values of Accuracy for simplification
    # You can customize this to use another method or metric for 'Best Model'
    best_models = results_df.apply(lambda x: x.str.split('±').str[0].astype(float).idxmax(), axis=1)
    results_df['Best Model'] = best_models

    return results_df


model_results = run_multiple_evaluations(X, y, DP.feature_names, n_runs=10)
model_results
dir_to_save = "./results/"

# Save ML score and feature_importance to CSV
model_results.to_csv(dir_to_save + "ML_model_scores.csv", index=False)

