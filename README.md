# MCAL-DTA
MCAL-DTA: A multimodal method integrating cross-attention and local feature enhancement for drug-target affinity prediction

# Requirements

pytorch==1.13
python==3.7.12
cuda==11.7
torch-geometric==2.3.1
mol2vec==0.1
rdkit==2023.3.2

Install [ESM2](https://github.com/facebookresearch/esm) from repo

``````
# Install PyTorch with CUDA support
pip install torch==1.13.0+cu117 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# Install PyTorch Geometric and dependencies
pip install torch-geometric==2.3.1
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
``````

For more requirements, please check the environment.yml file.

# Contents

## Project structure

    >  MCAL-DTA
       ├── data                                 - Data directory. The detailed information can be found in next section.
       ├── models                          
       │   ├── model_300dim.pkl                 - A pkl file of the pre-trained Mol2Vec model.
       ├── scripts                         
       │   ├── create_graph_data.py             - A python script used to generate drug substurcture graph data and protein graph data.
       │   ├── generate_drug_pretrain.py        - A python script used to generate drug pre-trained features.
       │   ├── generate_prot_pretrain.py        - A python script used to generate protein pre-trained features.
       ├── data_split.py                        - A python script for processing raw CSV data into datasets for 5-fold cross-validation experiments.
       ├── generate_contact_map.py              - A python script for processing raw CSV data into datasets for 5-fold cross-validation experiments.
       ├── main.py                              - A python script used to train the model under the warm and unseen-drug setting.
       ├── main_new_pair.py                     - A python script used to train the model under the unseen-pair setting.
       ├── main_new_prot.py                     - A python script used to train the model under the unseen-prot setting.
       ├── models.py                            - Original MCAL-DTA model file.
       ├── utils.py                             - A python script recording the various tools needed for training.

## Dataset

There are two DTA benchmark datasets were adopted in this work, including Davis and KIBA.

```text
>  data
    ├── davis / kiba                          - DTA dataset directory.
    │   ├── ligands_can.txt                   - A txt file recording ligands information (Original)
    │   ├── proteins.txt                      - A txt file recording proteins information (Original)
    │   ├── Y                                 - A file recording binding affinity score (Original)
    │   ├── folds                         
    │   │   ├── test_fold_setting1.txt        - A txt file recording test set entry (Original)
    │   │   └── train_fold_setting1.txt       - A txt file recording training set entry (Original)
    │   ├── (davis/kiba)_dict.txt             - A txt file recording the corresponding Uniprot ID for every protein in datasets (processed)
    │   ├── contact_map
    │   │   └── (Uniprot ID).npy              - A npy file recording the corresponding contact map for every protein in datasets (processed)
    │   ├── train.csv                         - Training set data in CSV format (processed)
    │   ├── test.csv                          - Test set data in CSV format (processed)
    │   ├── (davis/kiba)_iso_drug.csv         - Drug SMILES file processed according to isomeric SMILES.
```

For getting the contact map's npy file, you need to run the ``` generate_contact_map.py``` script, which would process the predicted protein structure file (pdb format) by AlphaFold2 into the npy file. 

For more details about contact map file, you could refer to [HiSIF-DTA](https://github.com/bixiangpeng/HiSIF-DTA).

# Example Usage

## Preprocess data

1. Getting the pre-trained features of drugs and proteins by running the ```generate_drug_pretrain.py``` and ```generate_prot_pretrain.py``` script.
2. Getting the graph features of drugs and proteins by running the ```create_graph_data.py``` script.
3. Running the ```data_split.py``` to get the training set, validation set and test set of 5-fold cross-validation experiments.

## Training model

4. Running ```main.py``` to train the model on Davis or KIBA dataset under the warm and unseen-drug setting.
5. Running ```main_new_prot.py``` and ```main_new_pair``` to train the model under the unseen-prot and unseen-pair setting.
