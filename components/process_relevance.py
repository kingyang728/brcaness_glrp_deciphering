#!python

import numpy as np
import pandas as pd
from multiprocessing import Pool
# from bioinfokit import visuz
#
# from scipy.sparse import csr_matrix
# #from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
# from sklearn.model_selection import train_test_split
#
# from components import data_handling #, glrp_scipy
#
# import time

rndm_state = 7
np.random.seed(rndm_state)


def stability_measure(fold_genes_list):
    N = len(fold_genes_list)
    similarities = [ProcessRelevances.jaccard_similarity(fold_genes_list[i], fold_genes_list[j])
                    for i in range(N) for j in range(N) if i < j]
    return np.sum(similarities)*2/(N*(N-1))

def get_folds_freq_genes_as_df(p, gene_list_set_diff):
    """Returns a dataframe of the most frequent genes withing folds subnetworks. The dataframe contains count."""
    fold_genes = [item for sublist in ProcessRelevances.get_top_genes_list(p.rel_df) for item in sublist]
    rest_genes = list(set(gene_list_set_diff).difference(set(fold_genes)))
    print(len(rest_genes))
    print(len(set(fold_genes)))
    my_dict = {i: fold_genes.count(i) for i in fold_genes}
    update_rest = [(g, 0) for g in rest_genes]
    my_dict.update(update_rest)
    frequent_genes_df = pd.DataFrame.from_dict(my_dict, orient = "index", columns=["Count"])
    frequent_genes_df.index.rename("Gene", inplace=True)
    # frequent_genes_df.sort_values(by="Count", ascending=False, inplace=True)
    frequent_genes_df.sort_index(inplace=True)
    # frequent_genes_df = frequent_genes_df.iloc[:ProcessRelevances.NUM_OF_VERTICES, ] # this select top 140 from the fold
    return frequent_genes_df


def get_freq_genes(p):
    """Gets a list of the most frequent genes withing folds subnetworks."""
    fold_genes = [item for sublist in ProcessRelevances.get_top_genes_list(p.rel_df) for item in sublist]
    my_dict = {i: fold_genes.count(i) for i in fold_genes}
    frequent_genes = set([key for index, key in enumerate(sorted(my_dict, reverse=True, key=my_dict.get)) if
                      index < ProcessRelevances.NUM_OF_VERTICES])
    return frequent_genes


def generate_fold_genes_most_frequent(proc_rel):
    with Pool(len(proc_rel)) as p:
        fold_genes_list = p.map(get_freq_genes, proc_rel)
    overal_union = set.union(*fold_genes_list)
    return fold_genes_list, overal_union

def generate_fold_genes_subnetwork_level(proc_rel, num_of_vertices=140):
    fold_genes_list = []
    overal_union = set()
    set_lengths = []
    for p in proc_rel:
        fold_genes = set([item for sublist in ProcessRelevances.get_top_genes_list(p.rel_df, num_of_vertices=num_of_vertices) for item in sublist])
        fold_genes_list.append(fold_genes)
        set_lengths.append(len(fold_genes))
        overal_union.update(fold_genes)
    print(set_lengths)
    print("mean len of subnetworks genes union over folds: {:5.2f}".format(np.mean(set_lengths)))
    return fold_genes_list, overal_union

def generate_fold_genes_test_set_level(proc_rel, num_of_top_vertices=140):
    fold_genes_list = []
    overal_union = set()
    for p in proc_rel:
        fold_genes = set(ProcessRelevances.get_top_features(p.rel_df, num_of_top_vertices))
        fold_genes_list.append(fold_genes)
        overal_union.update(fold_genes)
    return fold_genes_list, overal_union



class ProcessRelevances:

    NUM_OF_VERTICES = 140 # 600 BRCA GCNN

    def __init__(self, path_concordance, path_relevances):
        self.concordance_df = pd.read_csv(path_concordance)
        self.rel_df = pd.read_csv(path_relevances)
        #self.top_genes_concordant_patients = []

    def get_concordant_relevances_as_df(self):
        concordant_patients = self.concordance_df.loc[self.concordance_df["Concordance"] == 1, :]
        return self.rel_df[self.rel_df["Patient ID"].isin(concordant_patients["Patient ID"])]


    def get_concordant_relevances_as_df(self):
        concordant_patients = self.concordance_df.loc[self.concordance_df["Concordance"] == 1, :]
        return self.rel_df[self.rel_df["Patient ID"].isin(concordant_patients["Patient ID"])]

    def get_concordant_relevances_as_numpy(self):
        rel_df_concordant = self.get_concordant_relevances_as_df()
        return rel_df_concordant[rel_df_concordant.columns[1:]].to_numpy()

    def get_all_relevances_as_numpy(self):
        return self.rel_df[self.rel_df.columns[1:]].to_numpy()

    def get_all_labels_df(self):
        labels_df = pd.DataFrame(self.concordance_df["label"])
        labels_df = labels_df.set_index(self.concordance_df["Patient ID"])
        return labels_df

    def prepare_df_ranks_for_heatmap_on_genes(self, genes):
        """Prepares a data frame based on genes specified in genes variable."""
        rel_df_exp = self.rel_df[self.rel_df.columns[1:]]
        ranks = rel_df_exp.to_numpy().argsort().argsort()
        ranks = ranks/float(rel_df_exp.shape[1] - 1)
        rel_binary = pd.DataFrame(data=ranks, columns=rel_df_exp.columns, dtype=np.float64)
        rel_binary = rel_binary.set_index(self.rel_df["Patient ID"])
        rel_binary = rel_binary[genes]
        return rel_binary

    def prepare_df_binary_for_heatmap_on_genes(self, genes):
        """Prepares a data frame based on genes specified in genes variable."""
        rel_df_exp = self.rel_df[self.rel_df.columns[1:]]
        ranks = rel_df_exp.to_numpy().argsort().argsort()  # from 0 to 139 are top relevant genes
        ranks = rel_df_exp.shape[1] - ranks - 1 # from 0 to 139 are top relevant genes
        ranks[ranks < 140] = 1
        ranks[ranks >= 140] = 0
        rel_binary = pd.DataFrame(data=ranks, columns=rel_df_exp.columns, dtype=np.float64)
        rel_binary = rel_binary.set_index(self.rel_df["Patient ID"])
        rel_binary = rel_binary[genes]
        return rel_binary

    def prepare_df_for_heatmap_on_genes(self, genes=None):
        """Prepares a data frame based on genes specified in genes variable."""
        rel_values = self.rel_df[self.rel_df.columns[1:]]
        if genes:
            rel_values = self.rel_df[genes]
        else:
            rel_values = self.rel_df[self.rel_df.columns[1:]]
        rel_values = rel_values.set_index(self.rel_df["Patient ID"])
        return rel_values


    def prepare_df_for_heatmap(self):
        """
        Prepares a concordant data frame with genes that are union of the top relevant NUM_OF_VERTICES for each patient.
        """
        rel_values = self.get_concordant_relevances_as_df()
        rel_values = rel_values[rel_values.columns[1:]] # still need a data frame here
        rel_sums = rel_values.sum(axis=1)
        # rel_sums_df = pd.DataFrame(data={"Patient ID": rel_df_concordant["Patient ID"], "sum_of_rel": rel_sums})
        #
        num_of_top_vertices = self.NUM_OF_VERTICES
        top_relevances = []
        top_genes = set()
        for index, row in rel_values.iterrows():
            sorted_row = row.sort_values(ascending=False)[0:num_of_top_vertices]
            top_relevances.append([sorted_row.array])
            top_genes.update(sorted_row.index.tolist())

        top_genes = list(top_genes)
        print(len(top_genes))
        top_relevances_np = np.array(top_relevances).squeeze()
        top_relevances_percent = 100 * top_relevances_np.sum(axis=1) / rel_sums
        print(np.min(top_relevances_percent), np.max(top_relevances_percent))
        rel_values = rel_values[top_genes]
        rel_values = rel_values.set_index(self.get_concordant_relevances_as_df()["Patient ID"])
        rel_values = rel_values.transpose()
        rel_values.index.rename("Gene", inplace=True)
        return rel_values

    @staticmethod
    def rel_diff_simple(rel_1, rel_2):
        """Arguments are numpy arrays of the same shape."""
        rel_dif = rel_1 - rel_2
        absdiff = np.abs(rel_dif)
        N = absdiff.size
        print("rel_dif.shape: ",
              rel_dif.shape)
        print("error: max|rel_1 - rel_2|",
              np.max(absdiff))
        print("MSE: sum|rel_1 - rel_2|", np.sum(absdiff) / N)
        print("RMSE: between rel_1 and rel_2",
              np.sqrt(np.sum(absdiff * absdiff) / N))
        print("\n\n")

    @staticmethod
    def get_top_features(rel_values, num_of_top_vertices=NUM_OF_VERTICES):
        rel_values = rel_values[rel_values.columns[1:]]
        rel_sums = rel_values.sum(axis=0)
        rel_sums.sort_values(ascending=False, inplace=True)
        # print(rel_sums.head())
        # print(type(rel_sums))
        return rel_sums.index.tolist()[:num_of_top_vertices]

    @staticmethod
    def get_top_genes_list(rel_values, num_of_vertices = 140):
        """Getting the list of lists, where each element contain
        ProcessRelevances.NUM_OF_VERTICES top relevant genes corresponding to one patient."""
        rel_values = rel_values[rel_values.columns[1:]]
        # num_of_top_vertices = ProcessRelevances.NUM_OF_VERTICES
        num_of_top_vertices = num_of_vertices
        #top_relevances = []
        top_genes = []
        for index, row in rel_values.iterrows():
            sorted_row = row.sort_values(ascending=False)[0:num_of_top_vertices]
            # top_relevances.append([sorted_row.array])
            top_genes.append(sorted_row.index.tolist())
        return top_genes

    @staticmethod
    def jaccard_similarity(list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))

    @staticmethod
    def rel_diff_jaccard(process_rel_1, process_rel_2):
        """Arguments are of type ProcessRelevance. Calculates Jaccard distance for each pair of corresponding patients."""
        rel_1_df = process_rel_1.rel_df
        rel_2_df = process_rel_2.rel_df
        # rel_1_df = rel_1_df[rel_1_df["Patient ID"].isin(rel_2_df["Patient ID"])]
        # rel_2_df = rel_2_df[rel_2_df["Patient ID"].isin(rel_1_df["Patient ID"])]
        ids_equal = rel_1_df["Patient ID"].equals(rel_2_df["Patient ID"])
        print("\nrel_diff_jaccard function of ProcessRelevances class:")
        print("Correspondence of patients' IDs", ids_equal)
        if ids_equal:
            rel_1_df = rel_1_df.sort_values(by=["Patient ID"])
            rel_2_df = rel_2_df.sort_values(by=["Patient ID"])
            top_genes_1 = ProcessRelevances.get_top_genes_list(rel_1_df)
            top_genes_2 = ProcessRelevances.get_top_genes_list(rel_2_df)
            similarities = [ProcessRelevances.jaccard_similarity(top_genes_1[i], top_genes_2[i]) for i in range(len(top_genes_1))]
            # the genes lists are of the same length
            similarities_df = pd.DataFrame(data=similarities)
            similarities_df.set_index(rel_1_df["Patient ID"])
            return similarities_df, np.mean(similarities)
        else:
            print("Failed, patients IDs do not correspond")
            return None, 0.0

    @staticmethod
    def rel_diff_jaccard_concordant(process_rel_1, process_rel_2):
        """Arguments are of type ProcessRelevance. Calculates Jaccard distance for each pair of corresponding patients."""
        rel_1_df = process_rel_1.get_concordant_relevances_as_df()
        rel_2_df = process_rel_2.get_concordant_relevances_as_df()
        rel_1_df = rel_1_df[rel_1_df["Patient ID"].isin(rel_2_df["Patient ID"])]
        rel_2_df = rel_2_df[rel_2_df["Patient ID"].isin(rel_1_df["Patient ID"])]
        ids_equal = rel_1_df["Patient ID"].equals(rel_2_df["Patient ID"])
        print("\nrel_diff_jaccard_concordant function of ProcessRelevances class:")
        print("Correspondence of patients' IDs", ids_equal)
        if ids_equal:
            rel_1_df = rel_1_df.sort_values(by=["Patient ID"])
            rel_2_df = rel_2_df.sort_values(by=["Patient ID"])
            top_genes_1 = ProcessRelevances.get_top_genes_list(rel_1_df)
            top_genes_2 = ProcessRelevances.get_top_genes_list(rel_2_df)
            similarities = [ProcessRelevances.jaccard_similarity(top_genes_1[i], top_genes_2[i])
                            for i in range(len(top_genes_1))] # the genes lists are of the same length
            similarities_df = pd.DataFrame(data = similarities)
            similarities_df.set_index(rel_1_df["Patient ID"])
            return similarities_df, np.mean(similarities)
        else:
            print("Failed, patients IDs do not correspond")
            return None, 0.0

    @staticmethod
    def rel_diff(process_rel_1, process_rel_2):
        """Arguments are of type ProcessRelevance."""
        rel_1_df = process_rel_1.get_concordant_relevances_as_df()
        rel_2_df = process_rel_2.get_concordant_relevances_as_df()
        rel_1_df = rel_1_df[rel_1_df["Patient ID"].isin(rel_2_df["Patient ID"])]
        rel_2_df = rel_2_df[rel_2_df["Patient ID"].isin(rel_1_df["Patient ID"])]
        ids_equal = rel_1_df["Patient ID"].equals(rel_2_df["Patient ID"])
        print("\nrel_diff function of ProcessRelevances class:")
        print("Correspondence of patients' IDs", ids_equal)
        if ids_equal:
            rel_1_df = rel_1_df.sort_values(by=["Patient ID"])
            rel_2_df = rel_2_df.sort_values(by=["Patient ID"])
            ProcessRelevances.rel_diff_simple(rel_1_df[rel_1_df.columns[1:]].to_numpy(),
                                              rel_2_df[rel_2_df.columns[1:]].to_numpy())
        else:
            print("Failed, patients IDs do not correspond")