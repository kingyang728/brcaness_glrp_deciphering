import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, balanced_accuracy_score
from components import data_handling as dh, glrp_keras, nn_cnn_models
from lib import coarsening
import numpy as np
# from sklearn.metrics import
from scipy import interp
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
import joblib

import networkx as nx

from components.cheb_conv import ChebConvSlow, NonPos
from components.process_relevance import stability_measure


import time
import shap

import tensorflow as tf

def permute_selected_columns(X, idx):
    X_copy = X.copy()
    print("\tlen(idx):", len(idx))
    for i in idx:
        np.random.shuffle(X_copy[:, i])
    return X_copy


def print_accuracies(accuracies, f1_weighted_scores, name=""):
    n_splits = len(accuracies)
    ste_accuracy = np.std(accuracies, ddof=1) / np.sqrt(n_splits)
    ste_f1_weighted = np.std(f1_weighted_scores, ddof=1) / np.sqrt(n_splits)
    print("\n")
    print(name + ":")
    print("Metric:     \tmean \tstandard error")
    print("Accuracy:   \t%0.2f\t%0.2f" % (100 * np.mean(accuracies), 100 * ste_accuracy))
    print("F1_weighted:\t%0.2f\t%0.2f" % (100 * np.mean(f1_weighted_scores), 100 * ste_f1_weighted))


def write_accuracies_to_file(f, accuracies, f1_weighted_scores, name=""):
    n_splits = len(accuracies)
    ste_accuracy = np.std(accuracies, ddof=1) / np.sqrt(n_splits)
    ste_f1_weighted = np.std(f1_weighted_scores, ddof=1) / np.sqrt(n_splits)
    f.write(name + ":" + "\n")
    f.write("Metric:     \tmean \tstandard error\n")
    f.write("Accuracy:   \t%0.2f\t%0.2f\n" % (100 * np.mean(accuracies), 100 * ste_accuracy))
    f.write("F1_weighted:\t%0.2f\t%0.2f\n" % (100 * np.mean(f1_weighted_scores), 100 * ste_f1_weighted))
    f.write("\n")


def read_feature_selection_properties(dir_to_save):
    stabilities_df = pd.read_csv(dir_to_save + "stabilities.csv")
    print(stabilities_df.head())
    c_components_df = pd.read_csv(dir_to_save + "c_components.csv", header=None)
    print(c_components_df.head())
    f1_scores_top_df = pd.read_csv(dir_to_save + "f1_permute.csv", header=None)
    f1_scores_zero_df = pd.read_csv(dir_to_save + "f1_zeros.csv", header=None)
    data_dict = {"Stability": stabilities_df, "Connected components": c_components_df, "f1-score, feature values zero": f1_scores_zero_df, "f1_score, feature values permuted": f1_scores_top_df}
    # return [stabilities_df, c_components_df, f1_scores_top_df, f1_scores_zero_df]
    return data_dict
# def extract_values_for_plot(df):




def plot_the_training_keras_history(history):
    if history is not None:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    else:
        print("plotting the Keras training history: history is None")


def plot_the_gcnn_history(accuracy, loss, trained_losses):
    fig, ax1 = plt.subplots()
    ax1.plot(accuracy, 'b.-')
    lb1 = ax1.set_ylabel('validation AUC/accuracy', color='b')
    ax1.legend(["validation AUC"], loc=2)
    ax2 = ax1.twinx()
    ax2.plot(loss, 'r.-')
    ax2.plot(trained_losses, color='r', linestyle="--", marker='*')
    lb2 = ax2.set_ylabel('loss', color='r')
    ax2.legend(("validation loss", "training set loss"), loc=1)
    # fig.legend(("accuracy","validation loss", "training set loss"), labels=[lb1, lb2])
    # plt.figure(figsize=(18, 16))
    plt.show()


class KfoldCrossValidation:
    __dpi = 300

    def __init__(self, n_splits, name_of_ml_algorithm, shuffle=False, random_state=None):
        self.random_state = random_state
        self.n_splits = n_splits
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.accuracies = []
        self.f1_weighted_scores = []
        self.tprs = []
        self.aucs = []
        # self.fprs = []
        self.fpr_list = []
        self.tpr_list = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.mean_tpr = np.linspace(0, 1, 100)
        self.mean_auc = 0.5  # initial average of the area ander curve
        self.std_auc = 10.0  # initial std for the auc curve
        self.name = name_of_ml_algorithm
        self.number_of_performed_folds = 0  # the index to trek the number of folds

    def save_metrics(self, path_to_save=""):
        d = {"AUC": self.aucs, "Accuracy": self.accuracies, "F1_weighted": self.f1_weighted_scores}
        metrics_df = pd.DataFrame(d)
        metrics_df.to_csv(path_or_buf=path_to_save + "metrics_" + self.name + ".csv")

    def save_feature_selection_properties(self, feature_sizes, stabilities, c_components, f1_scores_top, f1_scores_zero, dir_to_save): # f1_scores_rand_zero, dir_to_save):
        stab_df = pd.DataFrame(data={"size": feature_sizes, "stabilities": stabilities})
        stab_df.to_csv(dir_to_save + "stabilities.csv", index=False)
        c_components_df = pd.DataFrame(c_components)
        c_components_df.to_csv(dir_to_save + "c_components.csv", index=False, header=False)
        f1_scores_top_df = pd.DataFrame(f1_scores_top)
        f1_scores_top_df.to_csv(dir_to_save + "f1_permute.csv", index=False, header=False)
        f1_scores_zero_df = pd.DataFrame(f1_scores_zero)
        f1_scores_zero_df.to_csv(dir_to_save + "f1_zeros.csv", index=False, header=False)
        # f1_scores_rand_zero_df = pd.DataFrame(f1_scores_rand_zero)
        # f1_scores_rand_zero_df.to_csv(dir_to_save + "f1_rand_zeros.csv", index=False, header=False)


    def read_feature_selection_properties(self, dir_to_save):
        stabilities_df = pd.read_csv(dir_to_save + "stabilities.csv", index=False)
        c_components_df = pd.read_csv(dir_to_save + "c_components.csv", index=False, header=False)
        f1_scores_top_df = pd.read_csv(dir_to_save + "f1_permute.csv", index=False, header=False)
        f1_scores_zero_df = pd.read_csv(dir_to_save + "f1_zeros.csv", index=False, header=False)
        return stabilities_df, c_components_df, f1_scores_top_df, f1_scores_zero_df

    #def wrt_true_label(self, y_test):

    def predict_and_compute_metrics(self, reconstructed_model, X_test, y_test, accuracies, f1_weighted_scores):
        y_preds = np.squeeze(reconstructed_model.predict(X_test))
        if len(y_preds.shape) > 1:
            y_preds = np.argmax(y_preds, axis=1)
        acc = accuracy_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds, average='weighted')
        accuracies.append(acc)
        f1_weighted_scores.append(f1)

    def run_random_forest_SHAP_10_fold_on_saved_models(self, X, y, feature_names, path_to_models, dir_to_save="./"):
        i = 1 # counter for models
        X = X - np.min(X)
        np.random.seed(self.random_state)
        k = 0
        accuracies = []
        f1_weighted_scores = []

        for train, test in self.cv.split(X, y):
            X_test = X[test, :]
            y_test = y[test]
            reconstructed_model = joblib.load(path_to_models + str(i))
            explainer = shap.TreeExplainer(reconstructed_model)
            shap_values = np.array(explainer.shap_values(X_test))
            aggregated_values = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
            print("aggregated values shape", aggregated_values.shape)
            aggregated_values = np.transpose(aggregated_values)
            print("aggregated values shape", aggregated_values.shape)

            # aggregated_values = coarsening.perm_data_back(aggregated_values, perm, len(feature_names))
            aggregated_values = np.squeeze(aggregated_values)
            # print("aggregated values shape", aggregated_values.shape)
            importances = pd.DataFrame(data={"Genes": feature_names,
                                             "importance": aggregated_values})
            importances = importances.sort_values(by="importance", ascending=False)
            # importances.to_csv(path_or_buf=dir_to_save + "test" + "_top_genes.csv", index=False)

            importances.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_genes_SHAP.csv", index=False)
            i = i + 1



    def run_random_forest_10_fold(self, X, y, feature_names, path_to_models, dir_to_save="./"):
        i = 1
        X = X - np.min(X)
        np.random.seed(self.random_state)
        k = 0
        accuracies = []
        f1_weighted_scores = []

        for train, test in self.cv.split(X, y):
            X_train = X[train, :]
            X_test = X[test, :]
            #     print(np.sum(np.isnan(X_test)))
            #     print(np.max(X_test))
            #     print(np.min(X_test))

            print(X_train.shape)
            print(X_test.shape)
            print(y[train].shape)

            # print(len(DP.adj_feature_graph.columns.tolist()))
            clf = RandomForestClassifier(n_estimators=10000, n_jobs=-1)
            clf.fit(X_train, y[train])
            joblib.dump(clf, path_to_models+str(i))

            # probas_ = clf.predict_proba(X_test)
            y_preds = clf.predict(X_test)
            acc = accuracy_score(y[test], y_preds)
            f1 = f1_score(y[test], y_preds, average='weighted')
            print("Fold: i = %d, test Accuraccy: %0.4f" % (i, acc))

            accuracies.append(acc)
            f1_weighted_scores.append(f1)
            importances = pd.DataFrame(data = {"Genes": feature_names,
                                               "importance": clf.feature_importances_}
                                       )
            importances = importances.sort_values(by = "importance", ascending=False)
            importances.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_genes_RF.csv", index=False)
            i += 1
            k += 1

        print_accuracies(accuracies, f1_weighted_scores, name="usual")

        # print("Metric\tmean\tstandard error")
        # # print("AUC: \t%0.4f\t%0.4f" % (self.mean_auc, self.ste_auc))
        # print("Accuracy:\t%0.4f\t%0.4f" % (np.mean(accuracies), self.ste_accuracy))
        # print("F1_weighted:\t%0.4f\t%0.4f" % (np.mean(self.f1_weighted_scores), self.ste_f1_weighted))

    def run_random_forest_10_fold_balanced_acc(self, X, y, feature_names, path_to_models, dir_to_save="./"):
        i = 1
        X = X - np.min(X)
        np.random.seed(self.random_state)
        k = 0
        accuracies = []
        f1_weighted_scores = []

        for train, test in self.cv.split(X, y):
            X_train = X[train, :]
            X_test = X[test, :]
            #     print(np.sum(np.isnan(X_test)))
            #     print(np.max(X_test))
            #     print(np.min(X_test))

            print(X_train.shape)
            print(X_test.shape)
            print(y[train].shape)

            # print(len(DP.adj_feature_graph.columns.tolist()))
            clf = RandomForestClassifier(n_estimators=10000, n_jobs=-1)
            clf.fit(X_train, y[train])
            joblib.dump(clf, path_to_models+str(i))

            # probas_ = clf.predict_proba(X_test)
            y_preds = clf.predict(X_test)
            acc = balanced_accuracy_score(y[test], y_preds)
            f1 = f1_score(y[test], y_preds, average='weighted')
            print("Fold: i = %d, test Accuraccy: %0.4f" % (i, acc))

            accuracies.append(acc)
            f1_weighted_scores.append(f1)
            importances = pd.DataFrame(data = {"Genes": feature_names,
                                               "importance": clf.feature_importances_}
                                       )
            importances = importances.sort_values(by = "importance", ascending=False)
            importances.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_genes_RF.csv", index=False)
            i += 1
            k += 1

        print("\t!!!")
        print("\tBalanced accuracy is computed")
        print_accuracies(accuracies, f1_weighted_scores, name="usual")


    def influence_of_features_fold_vise_on_saved_models_GCNN_10_fold(self, X, y, top_genes_dfs, feature_names, prior_network, path_to_models, dir_to_save="./", feature_sizes=[200], perm=None):
        X = X - np.min(X)
        np.random.seed(self.random_state)
        stabilities = np.zeros(shape=(len(feature_sizes),))
        f1_scores_top = np.zeros(shape=(self.n_splits, len(feature_sizes),))
        f1_scores_zero = np.zeros(shape=(self.n_splits, len(feature_sizes),))
        f1_weighted_scores_rand_z = np.zeros(shape=(self.n_splits, len(feature_sizes),))
        c_components = np.zeros(shape=(self.n_splits, len(feature_sizes),))

        for j, fs in enumerate(feature_sizes):
            top_genes_list = [df.iloc[0:fs, ]["Genes"].values for df in top_genes_dfs]
            stabilities[j] = 100 * stability_measure(top_genes_list)
            print("stability, {} genes set:                    {:5.2f}".format(fs,
                                                                               stabilities[j]))
        i = 1
        k = 0

        for train, test in self.cv.split(X, y):
            X_test = X[test, :]
            y_test = y[test]
            reconstructed_model = tf.keras.models.load_model(path_to_models + str(i),
                                                             custom_objects={"ChebConvSlow": ChebConvSlow,
                                                                             "NonPos": NonPos})
            for j, fs in enumerate(feature_sizes):
                top_genes = top_genes_dfs[k].iloc[0:fs, ]["Genes"].values

                G = prior_network.subgraph(top_genes)
                c_components[k, j] = nx.algorithms.components.number_connected_components(G)
                print("Connected components:", c_components[k, j])

                ind_bool = np.array(pd.Series(feature_names).isin(top_genes))
                ind_col = np.array(range(X.shape[1]))
                ind_col_top = ind_col[ind_bool]

                ind_col_rand = np.random.choice(X.shape[1], ind_col_top.shape[0], replace=False)

                # np.random.seed(self.random_state)
                # st2 = np.random.get_state()
                y_preds = np.squeeze(reconstructed_model.predict(coarsening.perm_data(X_test, perm)))
                y_preds = np.argmax(y_preds, axis=1)

                f1_true = f1_score(y_test, y_preds, average='weighted')
                print("\tf1_true", f1_true)

                X_test_zero = X_test.copy()
                X_test_zero[:, ind_col_top] = 0
                X_test_zero = coarsening.perm_data(X_test_zero, perm)
                y_preds = np.squeeze(reconstructed_model.predict(X_test_zero))
                y_preds = np.argmax(y_preds, axis=1)

                f1_scores_zero[k,j] = f1_score(y_test, y_preds, average='weighted')
                print("\tf1_scores_zero[k,j]:", f1_scores_zero[k, j])

                X_test_top = permute_selected_columns(X_test, ind_col_top)
                X_test_top = coarsening.perm_data(X_test_top, perm)
                y_preds = np.squeeze(reconstructed_model.predict(X_test_top))
                y_preds = np.argmax(y_preds, axis=1)

                f1_scores_top[k,j] = f1_score(y_test, y_preds, average='weighted')
                print("\tf1_scores_top[k,j]:", f1_scores_top[k,j])

                X_test_rand = permute_selected_columns(X_test, ind_col_rand)
                X_test_rand = coarsening.perm_data(X_test_rand, perm)
                y_preds = np.squeeze(reconstructed_model.predict(X_test_rand))
                y_preds = np.argmax(y_preds, axis=1)

                f1_rand = f1_score(y_test, y_preds, average='weighted')
                print("\tf1_rand", f1_rand)

                X_test_rand_zero = X_test.copy()
                X_test_rand_zero[:, ind_col_rand] = 0
                X_test_rand_zero = coarsening.perm_data(X_test_rand_zero, perm)
                y_preds = np.squeeze(reconstructed_model.predict(X_test_rand_zero))
                y_preds = np.argmax(y_preds, axis=1)

                f1_rand = f1_score(y_test, y_preds, average='weighted')
                f1_weighted_scores_rand_z[k,j] = f1_rand
                print("\tf1_rand_zero", f1_rand)

            i += 1
            k += 1

        self.save_feature_selection_properties(feature_sizes, stabilities, c_components, f1_scores_top, f1_scores_zero, # f1_weighted_scores_rand_z,
                                               dir_to_save)


    def influence_of_features_on_performance_fold_vise_on_saved_models_RF_10_fold(self, X, y, top_genes_dfs, feature_names, prior_network, path_to_models, dir_to_save="./", feature_sizes=[200]):
        """
        Write the feature properties (feature_sizes, stabilities, c_components, f1_scores_top, f1_scores_zero) into a file.
        """
        
        X = X - np.min(X)
        np.random.seed(self.random_state)
        stabilities = np.zeros(shape=(len(feature_sizes),))
        f1_scores_top = np.zeros(shape=(self.n_splits, len(feature_sizes),))
        f1_scores_zero = np.zeros(shape=(self.n_splits, len(feature_sizes),))
        c_components = np.zeros(shape=(self.n_splits, len(feature_sizes),))

        for j, fs in enumerate(feature_sizes):
            top_genes_list = [df.iloc[0:fs, ]["Genes"].values for df in top_genes_dfs]
            stabilities[j] = 100 * stability_measure(top_genes_list)
            print("stability, {} genes set:                    {:5.2f}".format(fs,
                                                                               stabilities[j]))
        i = 1
        k = 0

        for train, test in self.cv.split(X, y):
            X_test = X[test, :]
            y_test = y[test]
            reconstructed_model = joblib.load(path_to_models + str(i))

            for j, fs in enumerate(feature_sizes):
                top_genes = top_genes_dfs[k].iloc[0:fs, ]["Genes"].values

                G = prior_network.subgraph(top_genes)
                c_components[k, j] = nx.algorithms.components.number_connected_components(G)
                print("Connected components:", c_components[k, j])

                ind_bool = np.array(pd.Series(feature_names).isin(top_genes))
                print(ind_bool)
                ind_col = np.array(range(X.shape[1]))
                ind_col_top = ind_col[ind_bool]

                X_test_zero = X_test.copy()
                X_test_zero[:, ind_col_top] = 0
                y_preds = np.squeeze(reconstructed_model.predict(X_test_zero))
                # if len(y_preds.shape) > 1:
                #     y_preds = np.argmax(y_preds, axis=1)
                f1_scores_zero[k, j] = f1_score(y_test, y_preds, average='weighted')

                X_test_top = permute_selected_columns(X_test, ind_col_top)
                y_preds = np.squeeze(reconstructed_model.predict(X_test_top))
                f1_scores_top[k, j] = f1_score(y_test, y_preds, average='weighted')

            k += 1
            i += 1

        self.save_feature_selection_properties(feature_sizes, stabilities, c_components, f1_scores_top, f1_scores_zero, dir_to_save)



    def zeroing_and_permuting_selected_features_fold_vise_on_saved_models_RF_10_fold(self, X, y, top_genes_list, feature_names, path_to_models, dir_to_save="./"):
        i = 1
        X = X - np.min(X)
        np.random.seed(self.random_state)

        accuracies = []
        f1_weighted_scores = []
        accuracies_top = []
        f1_weighted_scores_top = []
        accuracies_rand = []
        f1_weighted_scores_rand = []

        accuracies_top_z = []
        f1_weighted_scores_top_z = []
        accuracies_rand_z = []
        f1_weighted_scores_rand_z = []

        # rec_models = []
        # for k in range(10):
        #     rec_models.append(tf.keras.models.load_model(path_to_models + str(i), custom_objects={"ChebConvSlow": ChebConvSlow, "NonPos": NonPos}))

        k = 0

        for train, test in self.cv.split(X, y):
            X_test = X[test, :]
            y_test = y[test]
            ind_bool = np.array(pd.Series(feature_names).isin(top_genes_list[k]))
            print(ind_bool)
            ind_col = np.array(range(X.shape[1]))
            ind_col_top = ind_col[ind_bool]
            print(ind_col_top.shape[0])
            ind_col_rand = np.random.choice(X.shape[1], ind_col_top.shape[0], replace=False)

            # np.random.seed(self.random_state)
            # st2 = np.random.get_state()

            X_test_top = X_test.copy()
            X_test_top[:, ind_col_top] = 0
            X_test_rand = X_test.copy()
            X_test_rand[:, ind_col_rand] = 0

            # np.random.set_state(st2)
            # print("\n\tRandom State, reseting seed", np.random.get_state()[1][:10], "\n")

            reconstructed_model = joblib.load(path_to_models + str(i))

            self.predict_and_compute_metrics(reconstructed_model, X_test, y_test, accuracies, f1_weighted_scores)
            print("Fold: i = %d, test Accuraccy: %0.4f" % (i, accuracies[k]))

            self.predict_and_compute_metrics(reconstructed_model, X_test_top, y_test, accuracies_top_z,
                                             f1_weighted_scores_top_z)
            print("Fold: i = %d, test top (zero) Accuraccy: %0.4f" % (i, accuracies_top_z[k]))

            self.predict_and_compute_metrics(reconstructed_model, X_test_rand, y_test, accuracies_rand_z,
                                             f1_weighted_scores_rand_z)
            print("Fold: i = %d, test rand (zero) Accuraccy: %0.4f" % (i, accuracies_rand_z[k]))

            X_test_top = permute_selected_columns(X_test, ind_col_top)
            X_test_rand = permute_selected_columns(X_test, ind_col_rand)
            # print("X_test.shape:", X_test.shape)

            self.predict_and_compute_metrics(reconstructed_model, X_test_top, y_test, accuracies_top, f1_weighted_scores_top)
            print("Fold: i = %d, test top Accuraccy: %0.4f" % (i, accuracies_top[k]))

            self.predict_and_compute_metrics(reconstructed_model, X_test_rand, y_test, accuracies_rand, f1_weighted_scores_rand)
            print("Fold: i = %d, test rand Accuraccy: %0.4f" % (i, accuracies_rand[k]))


            i += 1
            k += 1

            # shap_values = np.abs(e.shap_values(X_test)[0])
            # print(shap_values.min(), shap_values[0].max())

        print_accuracies(accuracies, f1_weighted_scores, name="usual")
        print_accuracies(accuracies_top_z, f1_weighted_scores_top_z, name="top_zero")
        print_accuracies(accuracies_rand_z, f1_weighted_scores_rand_z, name="rand_zero")

        print_accuracies(accuracies_top, f1_weighted_scores_top, name="top")
        print_accuracies(accuracies_rand, f1_weighted_scores_rand, name="rand")


        with open(dir_to_save + 'zeroing_and_usual_top_rand_acc_f1.txt', 'w') as f:
            write_accuracies_to_file(f, accuracies, f1_weighted_scores, name="usual")
            write_accuracies_to_file(f, accuracies_top_z, f1_weighted_scores_top_z, name="top_z")
            write_accuracies_to_file(f, accuracies_rand_z, f1_weighted_scores_rand_z, name="rand_z")
            write_accuracies_to_file(f, accuracies_top, f1_weighted_scores_top, name="top")
            write_accuracies_to_file(f, accuracies_rand, f1_weighted_scores_rand, name="rand")


    def zeroing_and_permuting_selected_features_fold_vise_on_saved_models_GCNN_10_fold(self, X, y, top_genes_list, feature_names, path_to_models, dir_to_save="./", perm=None):
        i = 1
        X = X - np.min(X)
        np.random.seed(self.random_state)

        accuracies = []
        f1_weighted_scores = []
        accuracies_top = []
        f1_weighted_scores_top = []
        accuracies_rand = []
        f1_weighted_scores_rand = []

        accuracies_top_z = []
        f1_weighted_scores_top_z = []
        accuracies_rand_z = []
        f1_weighted_scores_rand_z = []

        # rec_models = []
        # for k in range(10):
        #     rec_models.append(tf.keras.models.load_model(path_to_models + str(i), custom_objects={"ChebConvSlow": ChebConvSlow, "NonPos": NonPos}))

        k = 0

        for train, test in self.cv.split(X, y):
            X_test = X[test, :]
            y_test = y[test]
            ind_bool = np.array(pd.Series(feature_names).isin(top_genes_list[k]))
            print(ind_bool)
            ind_col = np.array(range(X.shape[1]))
            ind_col_top = ind_col[ind_bool]
            print(ind_col_top.shape[0])
            ind_col_rand = np.random.choice(X.shape[1], ind_col_top.shape[0], replace=False)

            # np.random.seed(self.random_state)
            # st2 = np.random.get_state()

            X_test_top = X_test.copy()
            X_test_top[:, ind_col_top] = 0
            X_test_rand = X_test.copy()
            X_test_rand[:, ind_col_rand] = 0
            X_test = coarsening.perm_data(X_test, perm)
            X_test_rand = coarsening.perm_data(X_test_rand, perm)
            X_test_top = coarsening.perm_data(X_test_top, perm)

            # np.random.set_state(st2)
            # print("\n\tRandom State, reseting seed", np.random.get_state()[1][:10], "\n")

            reconstructed_model = tf.keras.models.load_model(path_to_models + str(i), custom_objects={"ChebConvSlow": ChebConvSlow, "NonPos": NonPos})

            self.predict_and_compute_metrics(reconstructed_model, X_test, y_test, accuracies, f1_weighted_scores)
            print("Fold: i = %d, test Accuraccy: %0.4f" % (i, accuracies[k]))

            self.predict_and_compute_metrics(reconstructed_model, X_test_top, y_test, accuracies_top_z,
                                             f1_weighted_scores_top_z)
            print("Fold: i = %d, test top (zero) Accuraccy: %0.4f" % (i, accuracies_top_z[k]))

            self.predict_and_compute_metrics(reconstructed_model, X_test_rand, y_test, accuracies_rand_z,
                                             f1_weighted_scores_rand_z)
            print("Fold: i = %d, test rand (zero) Accuraccy: %0.4f" % (i, accuracies_rand_z[k]))

            X_test_top = permute_selected_columns(X_test, ind_col_top)
            X_test_rand = permute_selected_columns(X_test, ind_col_rand)
            # print("X_test.shape:", X_test.shape)
            X_test_rand = coarsening.perm_data(X_test_rand, perm)
            X_test_top = coarsening.perm_data(X_test_top, perm)

            self.predict_and_compute_metrics(reconstructed_model, X_test_top, y_test, accuracies_top, f1_weighted_scores_top)
            print("Fold: i = %d, test top Accuraccy: %0.4f" % (i, accuracies_top[k]))

            self.predict_and_compute_metrics(reconstructed_model, X_test_rand, y_test, accuracies_rand, f1_weighted_scores_rand)
            print("Fold: i = %d, test rand Accuraccy: %0.4f" % (i, accuracies_rand[k]))


            i += 1
            k += 1

            # shap_values = np.abs(e.shap_values(X_test)[0])
            # print(shap_values.min(), shap_values[0].max())

        print_accuracies(accuracies, f1_weighted_scores, name="usual")
        print_accuracies(accuracies_top_z, f1_weighted_scores_top_z, name="top_zero")
        print_accuracies(accuracies_rand_z, f1_weighted_scores_rand_z, name="rand_zero")

        print_accuracies(accuracies_top, f1_weighted_scores_top, name="top")
        print_accuracies(accuracies_rand, f1_weighted_scores_rand, name="rand")


        with open(dir_to_save + 'zeroing_and_usual_top_rand_acc_f1.txt', 'w') as f:
            write_accuracies_to_file(f, accuracies, f1_weighted_scores, name="usual")
            write_accuracies_to_file(f, accuracies_top_z, f1_weighted_scores_top_z, name="top_z")
            write_accuracies_to_file(f, accuracies_rand_z, f1_weighted_scores_rand_z, name="rand_z")
            write_accuracies_to_file(f, accuracies_top, f1_weighted_scores_top, name="top")
            write_accuracies_to_file(f, accuracies_rand, f1_weighted_scores_rand, name="rand")

    def zeroing_selected_features_fold_vise_on_saved_models_GCNN_10_fold(self, X, y, top_genes_list, feature_names, path_to_models, dir_to_save="./", perm=None):
        i = 1
        X = X - np.min(X)
        np.random.seed(self.random_state)
        k = 0
        accuracies = []
        f1_weighted_scores = []
        accuracies_top = []
        f1_weighted_scores_top = []
        accuracies_rand = []
        f1_weighted_scores_rand = []

        for train, test in self.cv.split(X, y):
            X_test = X[test, :]
            y_test = y[test]
            ind_bool = np.array(pd.Series(feature_names).isin(top_genes_list[k]))
            print(ind_bool)
            ind_col = np.array(range(X.shape[1]))
            ind_col_top = ind_col[ind_bool]
            print(ind_col_top.shape[0])
            ind_col_rand = np.random.choice(X.shape[1], ind_col_top.shape[0], replace=False)

            # np.random.seed(self.random_state)
            # st2 = np.random.get_state()

            X_test_top = X_test.copy()
            X_test_top[:, ind_col_top] = 0
            X_test_rand = X_test.copy()
            X_test_rand[:, ind_col_rand] = 0
            X_test = coarsening.perm_data(X_test, perm)
            X_test_rand = coarsening.perm_data(X_test_rand, perm)
            X_test_top = coarsening.perm_data(X_test_top, perm)

            # np.random.set_state(st2)
            # print("\n\tRandom State, reseting seed", np.random.get_state()[1][:10], "\n")

            reconstructed_model = tf.keras.models.load_model(path_to_models + str(i), custom_objects={"ChebConvSlow": ChebConvSlow, "NonPos": NonPos})

            self.predict_and_compute_metrics(reconstructed_model, X_test, y_test, accuracies, f1_weighted_scores)
            print("Fold: i = %d, test Accuraccy: %0.4f" % (i, accuracies[k]))

            self.predict_and_compute_metrics(reconstructed_model, X_test_top, y_test, accuracies_top, f1_weighted_scores_top)
            print("Fold: i = %d, test top Accuraccy: %0.4f" % (i, accuracies_top[k]))

            self.predict_and_compute_metrics(reconstructed_model, X_test_rand, y_test, accuracies_rand, f1_weighted_scores_rand)
            print("Fold: i = %d, test rand Accuraccy: %0.4f" % (i, accuracies_rand[k]))


            i += 1
            k += 1

            # shap_values = np.abs(e.shap_values(X_test)[0])
            # print(shap_values.min(), shap_values[0].max())

        print_accuracies(accuracies, f1_weighted_scores, name="usual")
        print_accuracies(accuracies_top, f1_weighted_scores_top, name="top")
        print_accuracies(accuracies_rand, f1_weighted_scores_rand, name="rand")

        with open(dir_to_save + 'zeroing_top_rand_acc_f1.txt', 'w') as f:
            write_accuracies_to_file(f, accuracies, f1_weighted_scores, name="usual")
            write_accuracies_to_file(f, accuracies_top, f1_weighted_scores_top, name="top")
            write_accuracies_to_file(f, accuracies_rand, f1_weighted_scores_rand, name="rand")



    def permute_selected_features_fold_vise_on_saved_models_GCNN_10_fold(self, X, y, top_genes_list, feature_names, path_to_models, dir_to_save="./", perm=None):
        i = 1
        X = X - np.min(X)
        np.random.seed(self.random_state)
        k = 0
        accuracies = []
        f1_weighted_scores = []
        accuracies_top = []
        f1_weighted_scores_top = []
        accuracies_rand = []
        f1_weighted_scores_rand = []

        for train, test in self.cv.split(X, y):
            # X_train, train_mean, train_std, non_zero_ind = dh.DataPreprocessor.normalize_data(X[train, :])
            # X_test = dh.DataPreprocessor.scale_data(X[test, :], train_mean, train_std, non_zero_ind)
            # X_train = X_train.astype(np.float32)
            # X_test = X_test.astype(np.float32)
            # X_train = X[train, :]
            X_test = X[test, :]
            # y_train = y[train]
            y_test = y[test]
            ind_bool = np.array(pd.Series(feature_names).isin(top_genes_list[k]))
            print(ind_bool)
            ind_col = np.array(range(X.shape[1]))
            ind_col_top = ind_col[ind_bool]
            print(ind_col_top.shape[0])
            ind_col_rand = np.random.choice(X.shape[1], ind_col_top.shape[0], replace=False)

            # np.random.seed(self.random_state)
            # st2 = np.random.get_state()

            X_test_top = permute_selected_columns(X_test, ind_col_top)
            X_test_rand = permute_selected_columns(X_test, ind_col_rand)
            X_test = coarsening.perm_data(X_test, perm)
            # print("X_test.shape:", X_test.shape)
            X_test_rand = coarsening.perm_data(X_test_rand, perm)
            X_test_top = coarsening.perm_data(X_test_top, perm)

            # np.random.set_state(st2)
            # print("\n\tRandom State, reseting seed", np.random.get_state()[1][:10], "\n")

            reconstructed_model = tf.keras.models.load_model(path_to_models + str(i), custom_objects={"ChebConvSlow": ChebConvSlow, "NonPos": NonPos})

            self.predict_and_compute_metrics(reconstructed_model, X_test, y_test, accuracies, f1_weighted_scores)
            # y_preds = np.squeeze(reconstructed_model.predict(X_test))
            # acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
            # f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
            # accuracies.append(acc)
            # f1_weighted_scores.append(f1)

            print("Fold: i = %d, test Accuraccy: %0.4f" % (i, accuracies[k]))

            self.predict_and_compute_metrics(reconstructed_model, X_test_top, y_test, accuracies_top, f1_weighted_scores_top)

            # y_preds = np.squeeze(reconstructed_model.predict(X_test_top))
            # labels_by_network = np.argmax(y_preds, axis=1)
            # acc = accuracy_score(y_test, labels_by_network)
            # f1 = f1_score(y_test, labels_by_network, average='weighted')
            # accuracies_top.append(acc)
            # f1_weighted_scores_top.append(f1)
            print("Fold: i = %d, test top Accuraccy: %0.4f" % (i, accuracies_top[k]))

            self.predict_and_compute_metrics(reconstructed_model, X_test_rand, y_test, accuracies_rand, f1_weighted_scores_rand)

            # y_preds = np.squeeze(reconstructed_model.predict(X_test_rand))
            # labels_by_network = np.argmax(y_preds, axis=1)
            # acc = accuracy_score(y_test, labels_by_network)
            # f1 = f1_score(y_test, labels_by_network, average='weighted')
            # accuracies_rand.append(acc)
            # f1_weighted_scores_rand.append(f1)
            print("Fold: i = %d, test rand Accuraccy: %0.4f" % (i, accuracies_rand[k]))

            i += 1
            k += 1

            # shap_values = np.abs(e.shap_values(X_test)[0])
            # print(shap_values.min(), shap_values[0].max())

        print_accuracies(accuracies, f1_weighted_scores, name="usual")
        print_accuracies(accuracies_top, f1_weighted_scores_top, name="top")
        print_accuracies(accuracies_rand, f1_weighted_scores_rand, name="rand")

        with open(dir_to_save + 'usual_top_rand_acc_f1.txt', 'w') as f:
            write_accuracies_to_file(f, accuracies, f1_weighted_scores, name="usual")
            write_accuracies_to_file(f, accuracies_top, f1_weighted_scores_top, name="top")
            write_accuracies_to_file(f, accuracies_rand, f1_weighted_scores_rand, name="rand")


    def select_glrp_features_label_wise_on_saved_models_GCNN_10_fold(self, X, y, params, feature_names, path_to_models, dir_to_save="./", perm=None):
        i = 1
        np.random.seed(self.random_state)
        X = X - np.min(X)
        X = coarsening.perm_data(X, perm)
        C = np.unique(y).size
        for train, test in self.cv.split(X, y):
            X_test = X[test, :]
            y_test = y[test]
            reconstructed_model = tf.keras.models.load_model(path_to_models + str(i), custom_objects={"ChebConvSlow": ChebConvSlow, "NonPos": NonPos})
            y_preds = np.squeeze(reconstructed_model.predict(X_test))
            y_preds = np.argmax(y_preds, axis=1)
            print("y_test.shape, y_preds.shape", y_test.shape, y_preds.shape)
            acc = accuracy_score(y_test, y_preds) # np.argmax(y_preds, axis=1))
            f1 = f1_score(y_test, y_preds, average='weighted') # np.argmax(y_preds, axis=1), average='weighted')
            print("\n\tFold: i = %d, test Accuraccy: %0.4f, test F1_weighted: %0.4f" % (i, acc, f1))


            C = np.unique(y).shape[0]
            I = np.eye(C)
            # y_train = I[y_train]
            y_hot_encoded = I[y_preds]

            glrp = glrp_keras.GraphLayerwiseRelevancePropagation(reconstructed_model, L=params["L"], K=params["K"],
                                                                 p=params["p"])

            # labels = np.zeros(shape=[X_test.shape[0], C])
            # # relevances_each_class = np.zeros(shape=[X_test.shape[0], len(DP.feature_names), C])
            importance = np.zeros(shape=[1, len(feature_names)], dtype=np.float32)
            rel = np.abs(glrp.get_relevances(X_test, y_hot_encoded))
            rel = coarsening.perm_data_back(rel, perm, len(feature_names))
            print("rel.shape", rel.shape)

            importance = rel.mean(axis=0)
            print("\n\tImportance:", importance.shape, type(importance[0]))
            importances = pd.DataFrame(data={"Genes": feature_names,
                                             "importance": importance})

            importances = importances.sort_values(by="importance", ascending=False)
            importances.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_genes_LRP.csv", index=False)
            i = i + 1


    def select_glrp_features_on_saved_models_GCNN_10_fold(self, X, y, params, feature_names, path_to_models, dir_to_save="./", perm=None):
        i = 1
        np.random.seed(self.random_state)
        X = X - np.min(X)
        X = coarsening.perm_data(X, perm)
        C = np.unique(y).size
        for train, test in self.cv.split(X, y):
            X_test = X[test, :]
            y_test = y[test]
            reconstructed_model = tf.keras.models.load_model(path_to_models + str(i), custom_objects={"ChebConvSlow": ChebConvSlow, "NonPos": NonPos})
            y_preds = np.squeeze(reconstructed_model.predict(X_test))

            acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
            f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
            print("\n\tFold: i = %d, test Accuraccy: %0.4f, test F1_weighted: %0.4f" % (i, acc, f1))

            glrp = glrp_keras.GraphLayerwiseRelevancePropagation(reconstructed_model, L=params["L"], K=params["K"],
                                                                 p=params["p"])

            labels = np.zeros(shape=[X_test.shape[0], C])
            # relevances_each_class = np.zeros(shape=[X_test.shape[0], len(DP.feature_names), C])
            importance = np.zeros(shape=[1, len(feature_names)], dtype=np.float32)
            for k in range(0, C):
                labels[:, k] = 1
                rel = np.abs(glrp.get_relevances(X_test, labels))
                labels[:, k] = 0
                rel = np.expand_dims(rel.mean(axis=0), axis=0)
                rel = coarsening.perm_data_back(rel, perm, len(feature_names))
                importance = importance + rel

            importance = np.squeeze(importance)
            print("\n\tImportance:", importance.shape, type(importance[0]))

            importances = pd.DataFrame(data={"Genes": feature_names,
                                             "importance": importance})

            importances = importances.sort_values(by="importance", ascending=False)
            importances.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_genes_LRP.csv", index=False)
            i = i + 1


    def select_shap_features_on_saved_models_GCNN_10_fold(self, X, y, feature_names, path_to_models, dir_to_save="./", perm=None):
        i = 1
        np.random.seed(self.random_state)
        X = X - np.min(X)
        X = coarsening.perm_data(X, perm)
        for train, test in self.cv.split(X, y):
            X_train = X[train, :]
            X_test = X[test, :]
            # y_train = y[train]
            y_test = y[test]
            reconstructed_model = tf.keras.models.load_model(path_to_models + str(i), custom_objects={"ChebConvSlow": ChebConvSlow, "NonPos": NonPos})
            y_preds = np.squeeze(reconstructed_model.predict(X_test))

            acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
            f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')



            print("\n\tFold: i = %d, test Accuraccy: %0.4f, test F1_weighted: %0.4f" % (i, acc, f1))

            if "MLP" in path_to_models:
                background = X_train
                samples = X_test
            else:
                background = np.expand_dims(X_train, axis=2)
                samples = np.expand_dims(X_test, axis=2)

            shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

            print("\n\n\t iteration:", i)
            print("\n\tif eager execution:", tf.executing_eagerly())

            print("\n\tStarting Shap")
            print("\n\tLength of the feature names", len(feature_names))
            print("background and test shapes", background.shape, X_test.shape)

            # !!!
            # Running Deep Explainer
            e = shap.DeepExplainer(reconstructed_model, background)
            print("\n\tCreated shap.DeepExplainer")

            start = time.time()
            print("\n\tStarting shap_values")
            shap_values = e.shap_values(samples, check_additivity=False)
            end = time.time()
            print("\n\tcreated shap_values")
            print("\n\tShap_values time:", end - start, "\n")
            print("\n\tshap values list", len(shap_values))
            print("\n\tshap values list to np", np.abs(shap_values).shape)

            aggregated_values = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
            #print("aggregated values shape", aggregated_values.shape)
            aggregated_values = np.transpose(aggregated_values)
            #print("aggregated values shape", aggregated_values.shape)

            aggregated_values = coarsening.perm_data_back(aggregated_values, perm, len(feature_names))
            aggregated_values = np.squeeze(aggregated_values)
            # print("aggregated values shape", aggregated_values.shape)
            importances = pd.DataFrame(data={"Genes": feature_names,
                                             "importance": aggregated_values})
            importances = importances.sort_values(by="importance", ascending=False)
            # importances.to_csv(path_or_buf=dir_to_save + "test" + "_top_genes.csv", index=False)

            importances.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_genes_SHAP.csv", index=False)
            i = i + 1


    def zeroing_feature_values_on_saved_models_GCNN_10_fold(self, X, y, top_genes, feature_names, dir_to_save="./", prefix="", perm=None):
        i = 1
        ind_bool = np.array(pd.Series(feature_names).isin(top_genes))
        print(ind_bool)
        ind_col = np.array(range(X.shape[1]))
        ind_col_top = ind_col[ind_bool]
        print(ind_col_top.shape[0])
        ind_col_rand = np.random.choice(X.shape[1], ind_col_top.shape[0], replace=False)

        np.random.seed(self.random_state)
        st2 = np.random.get_state()

        accuracies = []
        f1_weighted_scores = []
        accuracies_top = []
        f1_weighted_scores_top = []
        accuracies_rand = []
        f1_weighted_scores_rand = []

        X = X - np.min(X)
        for train, test in self.cv.split(X, y):
            # X_train, train_mean, train_std, non_zero_ind = dh.DataPreprocessor.normalize_data(X[train, :])
            # X_test = dh.DataPreprocessor.scale_data(X[test, :], train_mean, train_std, non_zero_ind)
            # X_train = X_train.astype(np.float32)
            # X_test = X_test.astype(np.float32)
            # X_train = X[train, :]
            X_test = X[test, :]
            # y_train = y[train]
            y_test = y[test]

            X_test_top = X_test.copy()
            X_test_top[:, ind_col_top] = 0
            X_test_rand = X_test.copy()
            X_test_rand[:, ind_col_rand] = 0
            # X_test_top = permute_selected_columns(X_test, ind_col_top)
            # X_test_rand = permute_selected_columns(X_test, ind_col_rand)
            X_test = coarsening.perm_data(X_test, perm)
            print("X_test.shape:", X_test.shape)
            X_test_rand = coarsening.perm_data(X_test_rand, perm)
            X_test_top = coarsening.perm_data(X_test_top, perm)

            np.random.set_state(st2)
            print("\n\tRandom State, reseting seed", np.random.get_state()[1][:10], "\n")

            reconstructed_model = tf.keras.models.load_model(dir_to_save + prefix + str(i))
            y_preds = np.squeeze(reconstructed_model.predict(X_test))

            acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
            f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
            accuracies.append(acc)
            f1_weighted_scores.append(f1)

            print("Fold: i = %d, test Accuraccy: %0.4f" % (i, acc))

            y_preds = np.squeeze(reconstructed_model.predict(X_test_top))
            labels_by_network = np.argmax(y_preds, axis=1)
            acc = accuracy_score(y_test, labels_by_network)
            f1 = f1_score(y_test, labels_by_network, average='weighted')
            accuracies_top.append(acc)
            f1_weighted_scores_top.append(f1)
            print("Fold: i = %d, test top Accuraccy: %0.4f" % (i, acc))

            y_preds = np.squeeze(reconstructed_model.predict(X_test_rand))
            labels_by_network = np.argmax(y_preds, axis=1)
            acc = accuracy_score(y_test, labels_by_network)
            f1 = f1_score(y_test, labels_by_network, average='weighted')
            accuracies_rand.append(acc)
            f1_weighted_scores_rand.append(f1)
            print("Fold: i = %d, test rand Accuraccy: %0.4f" % (i, acc))

            st2 = np.random.get_state()
            print("Fold: i = %d, test Accuraccy: %0.4f, test F1: %0.4f" % (i, acc, f1))
            i += 1

            # shap_values = np.abs(e.shap_values(X_test)[0])
            # print(shap_values.min(), shap_values[0].max())

        print_accuracies(accuracies, f1_weighted_scores, name="usual")
        print_accuracies(accuracies_top, f1_weighted_scores_top, name="top")
        print_accuracies(accuracies_rand, f1_weighted_scores_rand, name="rand")

        with open(dir_to_save + 'zeroing_usual_top_rand_acc_f1.txt', 'w') as f:
            write_accuracies_to_file(f, accuracies, f1_weighted_scores, name="usual")
            write_accuracies_to_file(f, accuracies_top, f1_weighted_scores_top, name="top")
            write_accuracies_to_file(f, accuracies_rand, f1_weighted_scores_rand, name="rand")


    def permute_feature_values_on_saved_models_GCNN_10_fold(self, X, y, top_genes, feature_names, path_to_models, dir_to_save="./", perm=None):
        i = 1
        ind_bool = np.array(pd.Series(feature_names).isin(top_genes))
        print(ind_bool)
        ind_col = np.array(range(X.shape[1]))
        ind_col_top = ind_col[ind_bool]
        print(ind_col_top.shape[0])
        ind_col_rand = np.random.choice(X.shape[1], ind_col_top.shape[0], replace=False)

        np.random.seed(self.random_state)
        st2 = np.random.get_state()

        accuracies = []
        f1_weighted_scores = []
        accuracies_top = []
        f1_weighted_scores_top = []
        accuracies_rand = []
        f1_weighted_scores_rand = []

        X = X - np.min(X)
        for train, test in self.cv.split(X, y):
            # X_train, train_mean, train_std, non_zero_ind = dh.DataPreprocessor.normalize_data(X[train, :])
            # X_test = dh.DataPreprocessor.scale_data(X[test, :], train_mean, train_std, non_zero_ind)
            # X_train = X_train.astype(np.float32)
            # X_test = X_test.astype(np.float32)
            # X_train = X[train, :]
            X_test = X[test, :]
            # y_train = y[train]
            y_test = y[test]
            X_test_top = permute_selected_columns(X_test, ind_col_top)
            X_test_rand = permute_selected_columns(X_test, ind_col_rand)
            X_test = coarsening.perm_data(X_test, perm)
            print("X_test.shape:", X_test.shape)
            X_test_rand = coarsening.perm_data(X_test_rand, perm)
            X_test_top = coarsening.perm_data(X_test_top, perm)

            np.random.set_state(st2)
            print("\n\tRandom State, reseting seed", np.random.get_state()[1][:10], "\n")

            reconstructed_model = tf.keras.models.load_model(path_to_models + str(i),  custom_objects={"ChebConvSlow": ChebConvSlow, "NonPos": NonPos})
            y_preds = np.squeeze(reconstructed_model.predict(X_test))

            acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
            f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
            accuracies.append(acc)
            f1_weighted_scores.append(f1)

            print("Fold: i = %d, test Accuraccy: %0.4f" % (i, acc))

            y_preds = np.squeeze(reconstructed_model.predict(X_test_top))
            labels_by_network = np.argmax(y_preds, axis=1)
            acc = accuracy_score(y_test, labels_by_network)
            f1 = f1_score(y_test, labels_by_network, average='weighted')
            accuracies_top.append(acc)
            f1_weighted_scores_top.append(f1)
            print("Fold: i = %d, test top Accuraccy: %0.4f" % (i, acc))

            y_preds = np.squeeze(reconstructed_model.predict(X_test_rand))
            labels_by_network = np.argmax(y_preds, axis=1)
            acc = accuracy_score(y_test, labels_by_network)
            f1 = f1_score(y_test, labels_by_network, average='weighted')
            accuracies_rand.append(acc)
            f1_weighted_scores_rand.append(f1)
            print("Fold: i = %d, test rand Accuraccy: %0.4f" % (i, acc))

            st2 = np.random.get_state()
            print("Fold: i = %d, test Accuraccy: %0.4f, test F1: %0.4f" % (i, acc, f1))
            i += 1

            # shap_values = np.abs(e.shap_values(X_test)[0])
            # print(shap_values.min(), shap_values[0].max())

        print_accuracies(accuracies, f1_weighted_scores, name="usual")
        print_accuracies(accuracies_top, f1_weighted_scores_top, name="top")
        print_accuracies(accuracies_rand, f1_weighted_scores_rand, name="rand")

        with open(dir_to_save + 'usual_top_rand_acc_f1.txt', 'w') as f:
            write_accuracies_to_file(f, accuracies, f1_weighted_scores, name="usual")
            write_accuracies_to_file(f, accuracies_top, f1_weighted_scores_top, name="top")
            write_accuracies_to_file(f, accuracies_rand, f1_weighted_scores_rand, name="rand")


    def permute_features_test_time_10_fold(self, top_genes, X, y, my_model, feature_names, dir_to_save="./"):
        i = 0
        ind_bool = np.array(pd.Series(feature_names).isin(top_genes))
        print(ind_bool)
        ind_col = np.array(range(X.shape[1]))
        ind_col_top = ind_col[ind_bool]
        print(ind_col_top.shape[0])
        ind_col_rand = np.random.choice(X.shape[1], ind_col_top.shape[0], replace=False)

        np.random.seed(self.random_state)
        st2 = np.random.get_state()

        accuracies = []
        f1_weighted_scores = []
        accuracies_top = []
        f1_weighted_scores_top = []
        accuracies_rand = []
        f1_weighted_scores_rand = []

        X = X - np.min(X)
        for train, test in self.cv.split(X, y):
            # X_train, train_mean, train_std, non_zero_ind = dh.DataPreprocessor.normalize_data(X[train, :])
            # X_test = dh.DataPreprocessor.scale_data(X[test, :], train_mean, train_std, non_zero_ind)
            # X_train = X_train.astype(np.float32)
            # X_test = X_test.astype(np.float32)
            X_train = X[train, :]
            X_test = X[test, :]
            y_train = y[train]
            y_test = y[test]
            X_test_top = permute_selected_columns(X_test, ind_col_top)
            X_test_rand = permute_selected_columns(X_test, ind_col_rand)
            print("X_train.shape:", X_train.shape)

            n_train = X_train.shape[0]
            # print(n_train)

            np.random.set_state(st2)
            print("\n\tRandom State, reseting seed", np.random.get_state()[1][:10], "\n")

            feature_number = X_train.shape[1]
            my_model.create(feature_number)
            history = my_model.fit(X_train, y_train, validation_data=[X_test, y_test], verbose=1)
            y_preds = my_model.predict(X_test)

            acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
            f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
            accuracies.append(acc)
            f1_weighted_scores.append(f1)

            print("Fold: i = %d, test Accuraccy: %0.4f" % (i, acc))

            labels_by_network = np.argmax(my_model.predict(X_test_top), axis=1)
            acc = accuracy_score(y_test, labels_by_network)
            f1 = f1_score(y_test, labels_by_network, average='weighted')
            accuracies_top.append(acc)
            f1_weighted_scores_top.append(f1)
            print("Fold: i = %d, test top Accuraccy: %0.4f" % (i, acc))

            labels_by_network = np.argmax(my_model.predict(X_test_rand), axis=1)
            acc = accuracy_score(y_test, labels_by_network)
            f1 = f1_score(y_test, labels_by_network, average='weighted')
            accuracies_rand.append(acc)
            f1_weighted_scores_rand.append(f1)
            print("Fold: i = %d, test rand Accuraccy: %0.4f" % (i, acc))
            st2 = np.random.get_state()
            print("Fold: i = %d, test Accuraccy: %0.4f, test F1: %0.4f" % (i, acc, f1))
            i += 1

            # shap_values = np.abs(e.shap_values(X_test)[0])
            # print(shap_values.min(), shap_values[0].max())

        print_accuracies(accuracies, f1_weighted_scores, name="usual")
        print_accuracies(accuracies_top, f1_weighted_scores_top, name="top")
        print_accuracies(accuracies_rand, f1_weighted_scores_rand, name="rand")

        with open(dir_to_save + 'usual_top_rand_acc_f1.txt', 'w') as f:
            write_accuracies_to_file(f, accuracies, f1_weighted_scores, name="usual")
            write_accuracies_to_file(f, accuracies_top, f1_weighted_scores_top, name="top")
            write_accuracies_to_file(f, accuracies_rand, f1_weighted_scores_rand, name="rand")

    def run_10_fold_save_models(self, X, y, params, to_get_model, patient_ids, feature_names, dir_to_save="./", L=None, K=None, p=None, perm=None):
        i = 0
        np.random.seed(self.random_state)
        X = X - np.min(X)
        X = coarsening.perm_data(X, perm)
        for train, test in self.cv.split(X, y):
            # X_train, train_mean, train_std, non_zero_ind = dh.DataPreprocessor.normalize_data(X[train, :])
            # X_test = dh.DataPreprocessor.scale_data(X[test, :], train_mean, train_std, non_zero_ind)
            # X_train = X_train.astype(np.float32)
            # X_test = X_test.astype(np.float32)
            X_train = X[train, :]
            X_test = X[test, :]
            y_train = y[train]
            y_test = y[test]

            #model = nn_cnn_models.get_bias_constrained_cheb_net_model(**params, mode_fast=False)
            model = to_get_model(**params, mode_fast=False)
            my_model = nn_cnn_models.MyChebNet(params, model)

            history = my_model.fit(X_train, y_train, validation_data=[X_test, y_test], verbose=1)
            y_preds = my_model.predict(X_test)

            acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
            f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
            self.accuracies.append(acc)
            self.f1_weighted_scores.append(f1)



            print("Fold: i = %d, test Accuraccy: %0.4f, test F1: %0.4f" % (i, acc, f1))
            i += 1

            my_model.model.save(dir_to_save+str(i), save_format='h5')

            # tf.keras.models.save_model(my_model.model, filepath=dir_to_save+str(i))


            # !!!
            # Explaining by LRP
            explain = False
            if explain:
                C = np.unique(y).shape[0]
                I = np.eye(C)
                # y_train = I[y_train]
                y_hot_encoded = I[y_test]
                glrp = glrp_keras.GraphLayerwiseRelevancePropagation(my_model.model,
                                                                     L=L, K=K, p=p)

                # # rel = glrp.get_relevances()[-1][:X_test.shape[0], :]
                rel = glrp.get_relevances(X_test, y_hot_encoded)
                print(type(rel))
                print(rel.shape)
                print(rel.sum(axis=1))
                if perm:
                    rel = coarsening.perm_data_back(rel, perm, len(feature_names))
                rel = np.squeeze(rel)
                patient_indexes_test = patient_ids[test]
                patient_ind_test_df = pd.DataFrame(data={"Patient ID": patient_indexes_test, "label": y_test})
                labels_by_network = np.argmax(y_preds, axis=1)
                print("labels_by_network type", labels_by_network.dtype)
                print("y_test type", y_test.dtype)
                concordance = y_test == labels_by_network

                concordance = concordance.astype(int)
                out_labels_conc_df = pd.DataFrame(np.array([labels_by_network, concordance]).transpose(),
                                                  columns=["Predicted", "Concordance"])
                concordance_df = patient_ind_test_df.join(out_labels_conc_df)
                concordance_df.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_predicted_concordance.csv",
                                      index=False)

                rel_df = pd.DataFrame(rel, columns=feature_names)
                rel_df = pd.DataFrame(data={"Patient ID": patient_indexes_test}).join(rel_df)
                rel_df.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_relevances_rendered_class.csv", index=False)

            # shap_values = np.abs(e.shap_values(X_test)[0])
            # print(shap_values.min(), shap_values[0].max())

        self.number_of_performed_folds = i
        # # print("last fpr shape", fpr.shape)
        # # print("fpr_list.len", len(self.fpr_list))
        # # print("Last thresholds", thresholds.shape)
        # # print("accuracy max %0.2f, min %0.2f, mean: %0.2f" % (np.max(self.accuracies), np.min(self.accuracies), np.mean(self.accuracies)))
        # self.mean_tpr = np.mean(self.tprs, axis=0)
        # self.mean_tpr[-1] = 1.0
        # self.mean_auc = auc(self.mean_fpr, self.mean_tpr)
        # self.std_auc = np.std(self.aucs, ddof=1)
        # self.ste_auc = self.std_auc/np.sqrt(self.n_splits)
        self.ste_accuracy = np.std(self.accuracies, ddof=1) / np.sqrt(self.n_splits)
        self.ste_f1_weighted = np.std(self.f1_weighted_scores, ddof=1) / np.sqrt(self.n_splits)
        print("Metric\tmean\tstandard error")
        # print("AUC: \t%0.4f\t%0.4f" % (self.mean_auc, self.ste_auc))
        print("Accuracy:\t%0.4f\t%0.4f" % (np.mean(self.accuracies), self.ste_accuracy))
        print("F1_weighted:\t%0.4f\t%0.4f" % (np.mean(self.f1_weighted_scores), self.ste_f1_weighted))

        with open(dir_to_save + 'usual_acc_f1.txt', 'w') as f:
            write_accuracies_to_file(f, self.accuracies, self.f1_weighted_scores, name="usual")


    def run_10_fold_generate_glrp_explanations(self, X, y, params, patient_ids, feature_names, path_to_models, dir_to_save="./", perm=None):
        i = 0
        np.random.seed(self.random_state)
        X = X - np.min(X)
        X = coarsening.perm_data(X, perm)
        for train, test in self.cv.split(X, y):
            # X_train, train_mean, train_std, non_zero_ind = dh.DataPreprocessor.normalize_data(X[train, :])
            # X_test = dh.DataPreprocessor.scale_data(X[test, :], train_mean, train_std, non_zero_ind)
            # X_train = X_train.astype(np.float32)
            # X_test = X_test.astype(np.float32)
            X_train = X[train, :]
            X_test = X[test, :]
            y_train = y[train]
            y_test = y[test]


            i += 1
            reconstructed_model = tf.keras.models.load_model(path_to_models + str(i),  custom_objects={"ChebConvSlow": ChebConvSlow, "NonPos": NonPos})
            reconstructed_model.summary()
            y_preds = np.squeeze(reconstructed_model.predict(X_test))

            acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
            f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
            self.accuracies.append(acc)
            self.f1_weighted_scores.append(f1)

            print("Fold: i = %d, test Accuraccy: %0.4f, test F1: %0.4f" % (i, acc, f1))
            # tf.keras.models.save_model(my_model.model, filepath=dir_to_save+str(i))

            # !!!
            # Explaining by LRP
            explain = True
            if explain:
                C = np.unique(y).shape[0]
                I = np.eye(C)
                # y_train = I[y_train]
                y_hot_encoded = I[y_test]
                glrp = glrp_keras.GraphLayerwiseRelevancePropagation(reconstructed_model, L=params["L"], #[0:2],
                                                                 K=params['K'], p=params['p'])

                # # rel = glrp.get_relevances()[-1][:X_test.shape[0], :]
                rel = glrp.get_relevances(X_test, y_hot_encoded)
                print(type(rel))
                print(rel.shape)
                print(rel.sum(axis=1))
                if perm:
                    rel = coarsening.perm_data_back(rel, perm, len(feature_names))
                rel = np.squeeze(rel)
                patient_indexes_test = patient_ids[test]
                patient_ind_test_df = pd.DataFrame(data={"Patient ID": patient_indexes_test, "label": y_test})
                labels_by_network = np.argmax(y_preds, axis=1)
                print("labels_by_network type", labels_by_network.dtype)
                print("y_test type", y_test.dtype)
                concordance = y_test == labels_by_network

                concordance = concordance.astype(int)
                out_labels_conc_df = pd.DataFrame(np.array([labels_by_network, concordance]).transpose(),
                                                  columns=["Predicted", "Concordance"])
                concordance_df = patient_ind_test_df.join(out_labels_conc_df)
                concordance_df.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_predicted_concordance.csv",
                                      index=False)

                rel_df = pd.DataFrame(rel, columns=feature_names)
                rel_df = pd.DataFrame(data={"Patient ID": patient_indexes_test}).join(rel_df)
                rel_df.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_relevances_rendered_class.csv", index=False)

            # shap_values = np.abs(e.shap_values(X_test)[0])
            # print(shap_values.min(), shap_values[0].max())

        self.number_of_performed_folds = i
        # # print("last fpr shape", fpr.shape)
        # # print("fpr_list.len", len(self.fpr_list))
        # # print("Last thresholds", thresholds.shape)
        # # print("accuracy max %0.2f, min %0.2f, mean: %0.2f" % (np.max(self.accuracies), np.min(self.accuracies), np.mean(self.accuracies)))
        # self.mean_tpr = np.mean(self.tprs, axis=0)
        # self.mean_tpr[-1] = 1.0
        # self.mean_auc = auc(self.mean_fpr, self.mean_tpr)
        # self.std_auc = np.std(self.aucs, ddof=1)
        # self.ste_auc = self.std_auc/np.sqrt(self.n_splits)
        self.ste_accuracy = np.std(self.accuracies, ddof=1) / np.sqrt(self.n_splits)
        self.ste_f1_weighted = np.std(self.f1_weighted_scores, ddof=1) / np.sqrt(self.n_splits)
        print("Metric\tmean\tstandard error")
        # print("AUC: \t%0.4f\t%0.4f" % (self.mean_auc, self.ste_auc))
        print("Accuracy:\t%0.4f\t%0.4f" % (np.mean(self.accuracies), self.ste_accuracy))
        print("F1_weighted:\t%0.4f\t%0.4f" % (np.mean(self.f1_weighted_scores), self.ste_f1_weighted))


    def select_features_with_shap(self, X, y, my_model, patient_ids, feature_names, dir_to_save="./"):
        i = 0
        np.random.seed(self.random_state)
        X = X - np.min(X)
        for train, test in self.cv.split(X, y):
            X_train = X[train, :]
            X_test = X[test, :]
            y_train = y[train]
            y_test = y[test]
            feature_number = X_train.shape[1]
            my_model.create(feature_number)
            history = my_model.fit(X_train, y_train, validation_data=[X_test, y_test], verbose=1)
            y_preds = my_model.predict(X_test)

            acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
            f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
            self.accuracies.append(acc)
            self.f1_weighted_scores.append(f1)

            print("Fold: i = %d, test Accuraccy: %0.4f, test F1: %0.4f" % (i, acc, f1))
            i += 1

            # background = X_train[np.random.choice(X_train.shape[0], 500, replace=False), :]

            # !!!
            # Explaining by SHAP
            explain = False
            if explain:
                background = np.expand_dims(X_train, axis=2)
                shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

                # explain predictions of the model on data_points
                # explainer = shap.DeepExplainer(my_model.predict, X_train)
                # shap_values = explainer.shap_values(X_test, nsamples=100)
                e = shap.DeepExplainer(my_model.model, background)
                import time
                start = time.time()
                shap_values = e.shap_values(np.expand_dims(X_test, axis=2), check_additivity=False)
                end = time.time()
                print("\n\tShap_values time:", end - start, "\n")
                print("aggregated values shape", shap_values.shape)


                aggregated_values = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
                print("aggregated values shape", aggregated_values.shape)


                # shap_values_processed = [np.squeeze(shap_values[l][i, :]) for i, l in np.ndenumerate(y_test)]
                # print(len(shap_values_processed))
                # print(shap_values_processed[5].shape)
                # shap_values_processed = np.array(shap_values_processed)
                # shap_values_processed[shap_values_processed < 0] = 0
                #
                # patient_indexes_test = patient_ids[test]
                # patient_ind_test_df = pd.DataFrame(data={"Patient ID": patient_indexes_test, "label": y_test})
                # labels_by_network = np.argmax(y_preds, axis=1)
                # print("labels_by_network type", labels_by_network.dtype)
                # print("y_test type", y_test.dtype)
                # concordance = y_test == labels_by_network
                #
                # concordance = concordance.astype(int)
                # out_labels_conc_df = pd.DataFrame(np.array([labels_by_network, concordance]).transpose(),
                #                                   columns=["Predicted", "Concordance"])
                # concordance_df = patient_ind_test_df.join(out_labels_conc_df)
                # concordance_df.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_predicted_concordance.csv",
                #                       index=False)
                #
                # rel_df = pd.DataFrame(shap_values_processed, columns=feature_names)
                # rel_df = pd.DataFrame(data={"Patient ID": patient_indexes_test}).join(rel_df)
                # rel_df.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_relevances_rendered_class.csv", index=False)

            # shap_values = np.abs(e.shap_values(X_test)[0])
            # print(shap_values.min(), shap_values[0].max())

        self.number_of_performed_folds = i
        # # print("last fpr shape", fpr.shape)
        # # print("fpr_list.len", len(self.fpr_list))
        # # print("Last thresholds", thresholds.shape)
        # # print("accuracy max %0.2f, min %0.2f, mean: %0.2f" % (np.max(self.accuracies), np.min(self.accuracies), np.mean(self.accuracies)))
        # self.mean_tpr = np.mean(self.tprs, axis=0)
        # self.mean_tpr[-1] = 1.0
        # self.mean_auc = auc(self.mean_fpr, self.mean_tpr)
        # self.std_auc = np.std(self.aucs, ddof=1)
        # self.ste_auc = self.std_auc/np.sqrt(self.n_splits)
        self.ste_accuracy = np.std(self.accuracies, ddof=1) / np.sqrt(self.n_splits)
        self.ste_f1_weighted = np.std(self.f1_weighted_scores, ddof=1) / np.sqrt(self.n_splits)
        print("Metric\tmean\tstandard error")
        # print("AUC: \t%0.4f\t%0.4f" % (self.mean_auc, self.ste_auc))
        print("Accuracy:\t%0.4f\t%0.4f" % (np.mean(self.accuracies), self.ste_accuracy))
        print("F1_weighted:\t%0.4f\t%0.4f" % (np.mean(self.f1_weighted_scores), self.ste_f1_weighted))


    def calculate_ROC_curves_with_shap(self, X, y, my_model, patient_ids, feature_names, dir_to_save="./"):
        i = 0
        np.random.seed(self.random_state)
        X = X - np.min(X)
        for train, test in self.cv.split(X, y):
            # X_train, train_mean, train_std, non_zero_ind = dh.DataPreprocessor.normalize_data(X[train, :])
            # X_test = dh.DataPreprocessor.scale_data(X[test, :], train_mean, train_std, non_zero_ind)
            # X_train = X_train.astype(np.float32)
            # X_test = X_test.astype(np.float32)
            X_train = X[train, :]
            X_test = X[test, :]
            y_train = y[train]
            y_test = y[test]
            feature_number = X_train.shape[1]
            my_model.create(feature_number)
            history = my_model.fit(X_train, y_train, validation_data=[X_test, y_test], verbose=1)
            y_preds = my_model.predict(X_test)

            acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
            f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
            self.accuracies.append(acc)
            self.f1_weighted_scores.append(f1)

            # Compute ROC curve and area the curve
            # fpr, tpr, thresholds = roc_curve(y_test, y_preds[:, 1])
            # self.tpr_list.append(tpr)
            # self.fpr_list.append(fpr)
            # self.tprs.append(interp(self.mean_fpr, fpr, tpr))
            # self.tprs[-1][0] = 0.0
            # roc_auc = auc(fpr, tpr)
            # print("Fold i = : ", i)
            # print("Fold: i = %d, test Accuraccy: %0.4f, test AUC: %0.4f" %(i, acc, roc_auc))
            # self.aucs.append(roc_auc)

            print("Fold: i = %d, test Accuraccy: %0.4f, test F1: %0.4f" % (i, acc, f1))
            i += 1

            # background = X_train[np.random.choice(X_train.shape[0], 500, replace=False), :]

            # !!!
            # Explaining by SHAP
            explain = False
            if explain:
                background = np.expand_dims(X_train, axis=2)
                shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

                # explain predictions of the model on data_points
                # explainer = shap.DeepExplainer(my_model.predict, X_train)
                # shap_values = explainer.shap_values(X_test, nsamples=100)
                e = shap.DeepExplainer(my_model.model, background)
                import time
                start = time.time()
                shap_values = e.shap_values(np.expand_dims(X_test, axis=2), check_additivity=False)
                end = time.time()
                print("\n\tShap_values time:", end - start, "\n")

                shap_values_processed = [np.squeeze(shap_values[l][i, :]) for i, l in np.ndenumerate(y_test)]
                print(len(shap_values_processed))
                print(shap_values_processed[5].shape)
                shap_values_processed = np.array(shap_values_processed)
                shap_values_processed[shap_values_processed < 0] = 0

                patient_indexes_test = patient_ids[test]
                patient_ind_test_df = pd.DataFrame(data={"Patient ID": patient_indexes_test, "label": y_test})
                labels_by_network = np.argmax(y_preds, axis=1)
                print("labels_by_network type", labels_by_network.dtype)
                print("y_test type", y_test.dtype)
                concordance = y_test == labels_by_network

                concordance = concordance.astype(int)
                out_labels_conc_df = pd.DataFrame(np.array([labels_by_network, concordance]).transpose(),
                                                  columns=["Predicted", "Concordance"])
                concordance_df = patient_ind_test_df.join(out_labels_conc_df)
                concordance_df.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_predicted_concordance.csv",
                                      index=False)

                rel_df = pd.DataFrame(shap_values_processed, columns=feature_names)
                rel_df = pd.DataFrame(data={"Patient ID": patient_indexes_test}).join(rel_df)
                rel_df.to_csv(path_or_buf=dir_to_save + "fold_" + str(i) + "_relevances_rendered_class.csv", index=False)

            # shap_values = np.abs(e.shap_values(X_test)[0])
            # print(shap_values.min(), shap_values[0].max())

        self.number_of_performed_folds = i
        # # print("last fpr shape", fpr.shape)
        # # print("fpr_list.len", len(self.fpr_list))
        # # print("Last thresholds", thresholds.shape)
        # # print("accuracy max %0.2f, min %0.2f, mean: %0.2f" % (np.max(self.accuracies), np.min(self.accuracies), np.mean(self.accuracies)))
        # self.mean_tpr = np.mean(self.tprs, axis=0)
        # self.mean_tpr[-1] = 1.0
        # self.mean_auc = auc(self.mean_fpr, self.mean_tpr)
        # self.std_auc = np.std(self.aucs, ddof=1)
        # self.ste_auc = self.std_auc/np.sqrt(self.n_splits)
        self.ste_accuracy = np.std(self.accuracies, ddof=1) / np.sqrt(self.n_splits)
        self.ste_f1_weighted = np.std(self.f1_weighted_scores, ddof=1) / np.sqrt(self.n_splits)
        print("Metric\tmean\tstandard error")
        # print("AUC: \t%0.4f\t%0.4f" % (self.mean_auc, self.ste_auc))
        print("Accuracy:\t%0.4f\t%0.4f" % (np.mean(self.accuracies), self.ste_accuracy))
        print("F1_weighted:\t%0.4f\t%0.4f" % (np.mean(self.f1_weighted_scores), self.ste_f1_weighted))


    def calculate_ROC_curves(self, X, y, my_model):
        i = 0
        for train, test in self.cv.split(X, y):
            X_train, train_mean, train_std, non_zero_ind = dh.DataPreprocessor.normalize_data(X[train, :])
            X_test = dh.DataPreprocessor.scale_data(X[test, :], train_mean, train_std, non_zero_ind)
            # X_train = X_train.astype(np.float32)
            # X_test = X_test.astype(np.float32)
            # X_train = X[train, :]
            # X_test = X[test, :]
            y_train = y[train]
            y_test = y[test]
            feature_number = X_train.shape[1]
            my_model.create(feature_number)
            history = my_model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1)
            y_preds = my_model.predict(X_test)

            acc = accuracy_score(y_test, np.argmax(y_preds, axis=1))
            f1 = f1_score(y_test, np.argmax(y_preds, axis=1), average='weighted')
            self.accuracies.append(acc)
            self.f1_weighted_scores.append(f1)

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_test, y_preds[:, 1])
            self.tpr_list.append(tpr)
            self.fpr_list.append(fpr)
            self.tprs.append(interp(self.mean_fpr, fpr, tpr))
            self.tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            # print("Fold i = : ", i)
            print("Fold: i = %d, test Accuraccy: %0.4f, test AUC: %0.4f" % (i, acc, roc_auc))
            self.aucs.append(roc_auc)
            i += 1

        self.number_of_performed_folds = i
        # print("last fpr shape", fpr.shape)
        # print("fpr_list.len", len(self.fpr_list))
        # print("Last thresholds", thresholds.shape)
        # print("accuracy max %0.2f, min %0.2f, mean: %0.2f" % (np.max(self.accuracies), np.min(self.accuracies), np.mean(self.accuracies)))
        self.mean_tpr = np.mean(self.tprs, axis=0)
        self.mean_tpr[-1] = 1.0
        self.mean_auc = auc(self.mean_fpr, self.mean_tpr)
        self.std_auc = np.std(self.aucs, ddof=1)
        self.ste_auc = self.std_auc / np.sqrt(self.n_splits)
        self.ste_accuracy = np.std(self.accuracies, ddof=1) / np.sqrt(self.n_splits)
        self.ste_f1_weighted = np.std(self.f1_weighted_scores, ddof=1) / np.sqrt(self.n_splits)
        print("Metric\tmean\tstandard error")
        print("AUC: \t%0.4f\t%0.4f" % (self.mean_auc, self.ste_auc))
        print("Accuracy:\t%0.4f\t%0.4f" % (np.mean(self.accuracies), self.ste_accuracy))
        print("F1_weighted:\t%0.4f\t%0.4f" % (np.mean(self.f1_weighted_scores), self.ste_f1_weighted))


    def save_ROC_curves(self, path_to_save=""):
        """Plot the information stored in properties of the class."""
        if self.number_of_performed_folds == 0:  # if loop with folds were not executed (in calculate_ROC_curves.)
            raise Exception("Nothing to plot, ROC curves were not calculated.")

        plt.figure(figsize=(10, 8))
        for i in range(0, self.number_of_performed_folds):
            plt.plot(self.fpr_list[i], self.tpr_list[i], lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, self.aucs[i]))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
        plt.plot(self.mean_fpr, self.mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (self.mean_auc, self.std_auc),
                 lw=2, alpha=.8)
        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(self.mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(self.mean_tpr - std_tpr, 0)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic ' + self.name)
        plt.legend(loc="lower right")
        plt.savefig(path_to_save + "ROC_" + self.name + ".png", dpi=self.__dpi)
        # plt.show()
