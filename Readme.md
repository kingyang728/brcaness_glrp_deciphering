# BRCAness Phenotype Deciphering using Graph Layer-wise Relevance Propagation (LRP)
This is analysis code of BRCAness Phenotype using the Layer-wise Relevance Propagation (LRP) method for [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375).
The code here is devoted to the paper [Deciphering BRCAness Phenotype in Cancer: A Graph Convolutional Neural Network Approach with Layer-wise Relevance Propagation Analysis](paperlink).
The version of the software that was published with the paper uses the Tensorflow 2.x and is under this [commit](https://gitlab.gwdg.de/MedBioinf/mtb/brcaness_glrp_deciphering).  

Current version of the code uses Tensorflow 2.x.
The implementation of LRP for Graph CNN is in the *components* folder.
The folder *lib* contains modifed by Hryhorii Chereda code from Michaël Defferrard's [Graph CNN](https://github.com/mdeff/cnn_graph) repository.

The file *run_glrp_ge_data_record_relevances.py* runs Graph Layer-wise Relevance Propagation (GLRP) to generate gene-wise relevances for individual TCGA-PanCancer patients. 
The details are in the [paper](paperlink).

    
## GLRP Requirements
To run the software one needs tensorflow, pandas, scipy, sklearn and matplotlib installed. I use docker, and the docker image can be built using the following content for the docker file:
<br>
<br>
FROM tensorflow/tensorflow:2.4.0-gpu  
RUN pip install pandas  
RUN pip install scipy  
RUN pip install sklearn  
RUN pip install matplotlib  

## Analysis Steps
1. **Download Data**: Place the required dataset in the `data` directory as per the instructions in `Readme.md` within the `data` folder.
2. **Label Samples**: Use `BRCAness_label.R` from the `LabelPreprocessing_and_Analysis` folder to label BRCAness and Non-BRCAness samples.
3. **Build Docker Image**: Build and run the Docker container for GLRP analysis:
    ```bash
    docker build -t brcaness_image .
    docker run -v /path/to/brcaness_glrp_deciphering-main:/brcaness_glrp_deciphering-main --gpus device=<GPU_ID> -it brcaness_image bash
    ```
    Navigate to the `brcaness_glrp_deciphering-main` directory and run the analysis:
    ```bash
    cd brcaness_glrp_deciphering-main/
    python run_glrp_ge_data_record_relevances.py
    ```
4. **Post-Processing**: After generating GLRP results in the `results` folder, execute:
    - `Top_Gene_Select_and_PathwayAnalysis.R` for gene selection and pathway analysis.
    - `Heatmap_Network.R` for visualization plots.
5. **Results**: Find the analysis plots in the `figures` folder.



## License
The source code is released under the terms of the MIT license
