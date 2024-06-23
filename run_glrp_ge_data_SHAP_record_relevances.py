#!python

"""
The script performs training of the Keras version of the GCNN model,
explaining data point-wise predictions, and aggregating data point-wise
explanations to deliver .
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split

from components import nn_cnn_models, data_handling#, glrp_keras, nn_cnn_evaluation
from lib import graph, coarsening
from sklearn.utils import class_weight
# from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import os

from components.cheb_conv import ChebConvSlow, NonPos

import time
import shap

rndm_state = 7
np.random.seed(rndm_state)

if __name__ == "__main__":

    path_to_feature_val = "./Data_EXP_LRPData/TCGA_BRCA/TCGA_exp_EXP.csv"
    path_to_feature_graph = "./Data_EXP_LRPData/TCGA_BRCA/TCGA_reactome_FIs.csv"
    path_to_labels = "./Data_EXP_LRPData/TCGA_BRCA/TCGA_BRCAness_Label.csv"

    # path_to_feature_val = "./Data_EXP_LRPData/GEO_metastatic_BRCA/GEO_HG_allwnt_1_component.csv"
    # path_to_feature_graph = "./Data_EXP_LRPData/GEO_metastatic_BRCA/allwnt_undirect_1_component.csv"
    # path_to_labels = "./Data_EXP_LRPData/GEO_metastatic_BRCA/labels_GEO_HG.csv"

    # path_to_feature_val = "./graph-lrp-master/Data_EXP_LRPData/GEO_HG_allwnt_1_component.csv"
    # path_to_feature_graph = "./graph-lrp-master/Data_EXP_LRPData/allwnt_undirect_1_component.csv"
    # path_to_labels = "./graph-lrp-master/Data_EXP_LRPData/TCGApanCancer_BRCAness_Label.csv"


    dir_to_save = "./results/TCGA_BRCAness/"
    model_name = "TCGA_BRCAness"

    if not os.path.exists(dir_to_save): os.makedirs(dir_to_save)

    DP = data_handling.DataPreprocessor(path_to_feature_values=path_to_feature_val, path_to_feature_graph=path_to_feature_graph,
                                        path_to_labels=path_to_labels)
    X = DP.get_feature_values_as_np_array()  # gene expression
    A = csr_matrix(DP.get_adj_feature_graph_as_np_array().astype(np.float32))  # adjacency matrix of the PPI network
    y = DP.get_labels_as_np_array()  # labels

    print("GE data, X shape: ", X.shape)
    print("Labels, y shape: ", y.shape)
    print("PPI network adjacency matrix, A shape: ", A.shape)

    X_train_unnorm, X_test_unnorm, y_train, y_test = train_test_split(X, y, test_size=0.10,
                                                                      stratify=y, random_state=rndm_state)

    # Need to know which patients got into train and test subsets
    _, _, patient_indexes_train, patient_indexes_test = train_test_split(X, DP.labels.columns.values.tolist(), test_size=0.10,
                                                                      stratify=y, random_state=rndm_state)

    # Data frame with test patients and corresponding ground truth labels
    patient_ind_test_df = pd.DataFrame(data={"Patient ID": patient_indexes_test, "label": y_test})

    # !!!
    # Making data lying in the interval [0, 8.35]
    X_train = X_train_unnorm - np.min(X)
    X_test = X_test_unnorm - np.min(X)

    print("X_train max", np.max(X_train))
    print("X_train min", np.min(X_train))
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train, shape: ", y_train.shape)
    print("y_test, shape: ", y_test.shape)

    # coarsening the PPI graph to perform pooling in the model
    graphs, perm = coarsening.coarsen(A, levels=2, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]

    X_train = coarsening.perm_data(X_train, perm)
    X_test = coarsening.perm_data(X_test, perm)

    n_train = X_train.shape[0]

    params = dict()
    params['dir_name']       = 'TCGA_BRCAness'
    params['num_epochs']     = 11
    # params['batch_size']     = 100
    params['batch_size']     = 109
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

    #model = models.cgcnn(L, **params)

    # !!!
    # Additional parameter, graph laplacian
    params["L"] = L

    model = nn_cnn_models.get_bias_constrained_cheb_net_model(**params, mode_fast=False)
    my_cheb_net_for_cv = nn_cnn_models.MyChebNet(params, model)

    start = time.time()
    my_cheb_net_for_cv.fit(x=np.expand_dims(X_train, 2), y=y_train, validation_data=[np.expand_dims(X_test, 2), y_test],
                           verbose=1)
    end = time.time()
    print("\n\tTraining time:", end-start, "\n")

    y_preds = my_cheb_net_for_cv.predict(np.expand_dims(X_test, 2))
    acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
    f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
    print("\n\nTest Accuraccy: %0.4f, test F1: %0.4f" % (acc, f1))

    # Saving model
    my_cheb_net_for_cv.model.save(dir_to_save + model_name, save_format='h5')

    # Loading model
    reconstructed_model = tf.keras.models.load_model(dir_to_save + model_name, custom_objects={"ChebConvSlow": ChebConvSlow, "NonPos": NonPos})

    # model_type = str(type(reconstructed_model)
    # print("\n\tModel_type:", model_type)
    # model_type = str(type(reloaded_model))
    # print("\n\tModel_type:", model_type)

    # y_preds = my_cheb_net_for_cv.predict(np.expand_dims(X_test, 2))

    # testing loaded model
    y_preds = reconstructed_model.predict(np.expand_dims(X_test, 2))
    acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
    f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
    print("Loaded model. Test Accuraccy: %0.4f, test F1: %0.4f" % (acc, f1))


###----------------------------------------------------------------START PART to CHANGE------------------


    # !!!
    # Preparing to run SHAP
    background = np.expand_dims(X_train, axis=2)
    # shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

    print("\nStarting Shap")

    # !!!
    # Running Deep Explainer
    #e = shap.DeepExplainer(my_cheb_net_for_cv.model, background)
    e = shap.GradientExplainer(reconstructed_model, background)
    print("\tCreated shap.GradientExplainer object")

    start = time.time()
    print("\tStarting shap_values")
    shap_values = e.shap_values(np.expand_dims(X_test, axis=2)) #, check_additivity=True)
    end = time.time()
    print("\tCreated shap_values")
    print("Shap_values. processing time:", end - start, "\n")
    print("\n\tShap values list length", len(shap_values))

    print("\n\tShap values list to np, shape", np.abs(shap_values).shape)

    aggregated_values = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
    print("\naggregated values shape", aggregated_values.shape)
    aggregated_values = np.transpose(aggregated_values)
    print("aggregated values shape", aggregated_values.shape)
###----------------------------------------------------------------END PART to CHANGE------------------

    # at this point the "aggregated_values.shape" has to be (1, 960)

    aggregated_values = coarsening.perm_data_back(aggregated_values, perm, X.shape[1])
    aggregated_values = np.squeeze(aggregated_values)
    print("aggregated values shape", aggregated_values.shape)
    importances = pd.DataFrame(data = {"Genes": DP.adj_feature_graph.columns.tolist(),
                                       "importance": aggregated_values})

    importances = importances.sort_values(by = "importance", ascending=False)
    importances.to_csv(path_or_buf=dir_to_save + "test_set" + "_top_genes.csv", index=False)


    print("\nThe last line of the code went through.")
