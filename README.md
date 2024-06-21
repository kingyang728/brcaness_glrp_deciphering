# BRCAness_GLRP_Deciphering


# Graph Layer-wise Relevance Propagation (GLRP) and BRCAness analysis
This is an implementation of the Layer-wise Relevance Propagation (LRP) method for [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375).
The code here is devoted to the paper [Deciphering BRCAness Phenotype in Cancer: A Graph Convolutional Neural Network Approach with Layer-wise Relevance Propagation Analysis](link).
The version of the software that was published with the paper uses the Tensorflow 1.x and is under this [commit](https://gitlab.gwdg.de/UKEBpublic/graph-lrp/-/tree/2bf6cdf8ff15eb1498bc60a607515ea43b89f135).  

Current version of the code uses Tensorflow 2.x.
The implementation of LRP for Graph CNN is in the *components* folder.
The folder *lib* contains modifed by me code from Michaël Defferrard's [Graph CNN](https://github.com/mdeff/cnn_graph) repository.
<!-- The visualization of the results can be found [on this website](http://mypathsem.bioinf.med.uni-goettingen.de/MetaRelSubNetVis). -->

The file *run_glrp_ge_data_record_relevances.py* runs Graph Layer-wise Relevance Propagation (GLRP) to generate gene-wise relevances for individual breast cancer patients. 
The details are in the [paper]().

The file *run_glrp_grid_mnist.py* executes training of GCNN on the MNIST data and applies GLRP to it.
    
## Requirements
To run the software one needs tensorflow, pandas, scipy, sklearn and matplotlib installed. I use docker, and the docker image can be built using the following content for the docker file:
<br>
<br>
FROM tensorflow/tensorflow:2.4.0-gpu  
RUN pip install pandas  
RUN pip install scipy  
RUN pip install sklearn  
RUN pip install matplotlib  

## TCGA-BRCA Data
The preprocessed breast cancer data is under this [link](). It contains three zip-archived csv files:  
Gene expression  *TCGA_exp_EXP.csv*  
Adjacency matrix *TCGA_reactome_FIs.csv*  
Patient labels *TCGA_BRCAness_Label.csv*  
Stratified cancer type labels *CancerEntity_combLabel.csv
To run the code, download the csv files into *GLRP_BRCAness/Data_EXP_LRPData/TCGA_BRCA* directory.


## License
The source code is released under the terms of the MIT license

