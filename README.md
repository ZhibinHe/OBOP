# One Brain, Open-Set Phenotypes (OBOP)

This repository contains the official implementation of the **OBOP Framework** (One Brain, Open-set Phenotype: A Semantic-Guided Model for Brain-Phenotype Prediction). Our framework establishes a generative paradigm for **zero shot brain phenotype prediction**. By bridging textual semantic definitions with neural representations, this framework transforms static and label dependent prediction into a dynamic and generalizable inference system. It enables effective cross task and cross domain knowledge transfer without requiring target domain training data.

## Key Features
* **Zero Shot Prediction:** The model dynamically generates predictors for open set (unseen) phenotypes based purely on their semantic text descriptions.
* **Cross Domain Transfer:** The framework trains on a source dataset (e.g., ABCD) and generalizes directly to independent target datasets (e.g., HCP A, HCP D, HCP YA).
* **Intra Dataset Transfer:** The model generalizes from seen phenotypes to unseen phenotypes within the same dataset cohort.
* **Multi Source Collaborative Training:** The framework combines multiple large scale datasets to enhance model generalization.
* **Biological Interpretability:** The model generates biologically valid Phenotype Activation Maps (PAMs) for arbitrary semantic descriptions.

---

## Repository Structure

* `10-preprocessing_Image_data.py`: Automates the creation of functional connectivity matrices (Pearson and Partial Correlation) from fMRI time series data and packages them into HDF5 (`.h5`) formats.
* `11-train_ABCD_test_CrossDomain_transfer.py`: Single source cross domain transfer (e.g., Train on ABCD $\rightarrow$ Test on HCP A).
* `12-train_Intra_test_CrossDomain_transfer.py`: Intra dataset transfer (Train on seen tasks $\rightarrow$ Test on unseen tasks within the same dataset).
* `13-generate_intra_phenotype_value_Semantic_Leakage.py`: Utility script to filter phenotype datasets and strictly control for semantic leakage.
* `13-train_Intra_test_Semantic_Leakage.py`: Intra dataset transfer evaluated under strict semantic leakage control.
* `14-train_Multi_test_CrossDomain_transfer.py`: Multi source collaborative training (e.g., Train on ABCD + HCP A + HCP D $\rightarrow$ Test on HCP YA).
* `15-train_ABCD_test_CrossDomain_transfer_Semantic_Sensitivity.py`: Evaluates model performance across different levels of semantic granularity.
* `16-train_ABCD_test_Nonsense.py`: Sanity check module that feeds biologically invalid text to verify the out of distribution rejection capabilities.

---


## Installation & Requirements

Ensure you have Python 3.8+ installed.

```bash
# Clone the repository
git clone [https://github.com/ZhibinHe/OBOP.git](https://github.com/ZhibinHe/OBOP.git)
cd OBOP
pip install -r requirements.txt


## Data Preparation & Structure
Data Access Notice: Due to strict data use agreements, we cannot provide the raw neuroimaging or phenotype datasets. Users must independently apply for access to the ABCD, HCP A, HCP D, and HCP YA datasets through their respective official data sharing platforms.

Once you obtain the data, you must organize it strictly according to the following directory structure to ensure compatibility with our scripts. The root directory for all data must reside at `/data/hzb/project/OBOP_data/`.

* Required Directory Tree (Using ABCD as an example)

```bash
/data/hzb/project/OBOP_data/
└── ABCD/
    ├── ABCD_subject_id/               # Subject ID lists (Plain text, one ID per line)
    │   ├── Subject_ID.txt             # Full cohort IDs
    │   ├── Subject_ID_train.txt       # Training set IDs (For Intra dataset tasks)
    │   └── Subject_ID_test.txt        # Testing set IDs (For Intra dataset tasks)
    │
    ├── ABCD_vol_mean/                 # Raw fMRI time series (Input for script 10)
    │   ├── sub-NDARINV003RTV85.txt
    │   ├── sub-NDARINV005V652G.txt
    │   └── ...
    │
    └── ABCD_phenotype/                # Phenotype scores and semantic data
        ├── ABCD_Phenotype_Value.csv   # Ground truth labels. First column MUST be 'NAME'
        ├── ABCD_Phenotype_Description.csv # Semantic descriptions (Requires a 'var_name' column)
        └── ABCD_Text_Features.pt      # Extracted semantic embeddings (See note below)
```

* Important Note on Semantic Text Features
Our framework relies on high dimensional text embeddings to guide the generative process. **You must independently generate these semantic features prior to training**. We recommend using the **BiomedCLIP** model to extract embeddings from the texts in your ``_Phenotype_Description.csv`` files. Save the resulting PyTorch tensors as ``.pt`` files within the respective phenotype directories. Ensure the tensor shape aligns perfectly with the number of prediction tasks (``[num_tasks, embedding_dimension]``).



## Using Pre trained Models (Hugging Face)

We provide pre trained weights for our models (Stage 1 Encoder and Stage 2 Decoder) on Hugging Face to facilitate immediate testing and reproducibility.

Download the weights from Hugging Face: ``https://huggingface.co/zhibinhe/OBOP``

1. Download the ``model_merged`` folder from the Hugging Face repository.

2. Place the ``model_merged`` folder in the following directory: ``/data/hzb/project/OBOP/model_merged/``.

3. Use the ``--test_only`` flag in the scripts below to run evaluations using these pre trained weights without retraining.



## Usage Instructions

The framework consists of two stages: a Supervised Encoder (Stage 1) and a Generative Decoder (Stage 2). The provided scripts automatically handle the transition between these stages.


* Cross Domain Transfer (Single Source)
Train on a source dataset and test on a completely independent target dataset.

```bash
# Train on ABCD and Test on HCP A
python 11-train_ABCD_test_CrossDomain_transfer.py --source_dataset ABCD --target_dataset HCP_A --zero_num_prediction 34 --gpu_id 0

# Test Only (requires pre trained weights in /data/hzb/project/OBOP/model_merged/)
python 11-train_ABCD_test_CrossDomain_transfer.py --test_only --source_dataset ABCD --target_dataset HCP_A --gpu_id 0

```

* Multi Source Collaborative Training
Train collaboratively on multiple datasets to improve the robustness of the generated parameter space.


```bash
# Train on ABCD, HCP A, and HCP D, then test on HCP YA
python 14-train_Multi_test_CrossDomain_transfer.py --source_dataset ABCD,HCP_A,HCP_D --target_dataset HCP_YA --gpu_id 0

```

* Intra Dataset Generalization
Train on a subset of known phenotypes and predict strictly unseen phenotypes within the same dataset.

```bash
# Train and Test within HCP A
python 12-train_Intra_test_CrossDomain_transfer.py --dataset HCP_A_intra --num_prediction 22 --zero_num_prediction 12 --gpu_id 0

```


* Semantic Leakage Control
To rigorously prove that the model relies on semantic mapping rather than data leakage, use the specialized script that filters the dataset.

```bash
# Step 1: Generate filtered phenotype values
python 13-generate_intra_phenotype_value_Semantic_Leakage.py --name Subject_ID_train.txt --description ABCD_Train_Phenotype_Description_SL.csv --value ABCD_Phenotype_Value.csv --out ABCD_Train_Phenotype_Value_SL.csv

# Step 2: Run the intra dataset prediction with strict leakage control
python 13-train_Intra_test_Semantic_Leakage.py --dataset ABCD_intra_SL --num_prediction 76 --zero_num_prediction 39 --gpu_id 0

```

* Biological Sanity Check (Nonsense Input)
Feed the model with abstract noise sequences or nonsense descriptions to verify that it outputs random noise maps rather than hallucinating false brain patterns.

```bash
python 16-train_ABCD_test_Nonsense.py --source_dataset ABCD --target_dataset HCP_YA --test_only --gpu_id 0

```

* Phenotype Activation Map (PAM) Surface Visualization

To interpret the biological validity of the generated features, you can visualize the Phenotype Activation Maps (PAMs) directly on a 3D cortical surface.

Navigate to the ``./PAM_Vis/`` directory.

Open MATLAB and run the ``map_PMA_2_Surface.m`` script.

The script maps the generated region level PAM weights onto the ``S1200`` standard cortical surface and exports them as ``.vtk`` files (e.g., ``surf_l_pam_HCP_A.vtk``).

You can open these generated ``.vtk`` files using standard 3D neuroimaging software, such as ParaView, to view the activation networks.

