"""
================================================================================
SCRIPT: Generate Filtered Phenotype Values (Internal/Semantic Leakage Control)
================================================================================
DESCRIPTION:
    This script is designed to filter phenotype data based on a specific list
    of subjects and target variables. It is used to prepare datasets for
    intra-dataset experiments while ensuring data consistency. It automatically
    identifies the dataset type (ABCD, HCP_A, HCP_D, HCP_YA) based on the input
    filenames and constructs the appropriate system paths.

USAGE:
    python generate_intra_phenotype_value_Semantic_Leakage.py \
        --name Subject_ID_train.txt \
        --description ABCD_Train_Phenotype_Description.csv \
        --value ABCD_Phenotype_Value.csv \
        --out ABCD_Train_Phenotype_Value.csv

    python generate_intra_phenotype_value_Semantic_Leakage.py \
        --name Subject_ID_train.txt \
        --description ABCD_Train_Phenotype_Description_SL.csv \
        --value ABCD_Phenotype_Value.csv \
        --out ABCD_Train_Phenotype_Value_SL.csv

    python generate_intra_phenotype_value_Semantic_Leakage.py \
        --name Subject_ID_test.txt \
        --description ABCD_Test_Phenotype_Description_SL.csv \
        --value ABCD_Phenotype_Value.csv \
        --out ABCD_Test_Phenotype_Value_SL.csv


INPUTS:
    1. --name (txt): A list of Subject IDs (e.g., Subject_ID_train.txt).
       Path: /data/hzb/project/OBOP_data/[DATASET]/[DATASET]_subject_id/
    2. --description (csv): Contains target 'var_name' columns to be extracted.
       Path: /data/hzb/project/OBOP_data/[DATASET]/[DATASET]_phenotype/
    3. --value (csv): The raw large phenotype data file containing all values.
       Path: /data/hzb/project/OBOP_data/[DATASET]/[DATASET]_phenotype/

OUTPUTS:
    - A filtered CSV file (--out) containing the selected subjects (rows)
      and selected phenotype variables (columns), with 'NAME' as the first column.
================================================================================
"""

import pandas as pd
import os
import argparse



# python generate_intra_phenotype_value_Semantic_Leakage.py --name Subject_ID_train.txt --description ABCD_Train_Phenotype_Description_SL.csv --value ABCD_Phenotype_Value.csv --out ABCD_Train_Phenotype_Value_SL.csv
# python generate_intra_phenotype_value_Semantic_Leakage.py --name Subject_ID_test.txt --description ABCD_Test_Phenotype_Description_SL.csv --value ABCD_Phenotype_Value.csv --out ABCD_Test_Phenotype_Value_SL.csv
# python generate_intra_phenotype_value_Semantic_Leakage.py --name Subject_ID_train.txt --description HCP_YA_Train_Phenotype_Description_SL.csv --value HCP_YA_Phenotype_Value.csv --out HCP_YA_Train_Phenotype_Value_SL.csv
# python generate_intra_phenotype_value_Semantic_Leakage.py --name Subject_ID_test.txt --description HCP_YA_Test_Phenotype_Description_SL.csv --value HCP_YA_Phenotype_Value.csv --out HCP_YA_Test_Phenotype_Value_SL.csv





def get_args():
    parser = argparse.ArgumentParser(description="Generate filtered phenotype value files for specific datasets.")
    parser.add_argument('--name', type=str, required=True, help='Filename of the Subject ID txt (e.g., Subject_ID_train.txt)')
    parser.add_argument('--description', type=str, required=True, help='Filename of the description csv')
    parser.add_argument('--value', type=str, required=True, help='Filename of the raw phenotype value csv')
    parser.add_argument('--out', type=str, required=True, help='Filename of the output csv')
    return parser.parse_args()

def generate_train_phenotype_value():
    args = get_args()

    # Automatically identify the dataset type to determine the directory structure
    # Supports ABCD, HCP_A, HCP_D, HCP_YA
    dataset_key = "ABCD"
    for key in ["HCP_A", "HCP_D", "HCP_YA"]:
        if key in args.name or key in args.description:
            dataset_key = key
            break

    # Define root paths for TXT and CSV files according to project structure
    txt_root = f"/data/hzb/project/OBOP_data/{dataset_key}/{dataset_key}_subject_id"
    csv_root = f"/data/hzb/project/OBOP_data/{dataset_key}/{dataset_key}_phenotype"

    # Construct full input file paths
    name_path = os.path.join(txt_root, args.name)
    desc_path = os.path.join(csv_root, args.description)
    val_path = os.path.join(csv_root, args.value)
    out_path = os.path.join(csv_root, args.out)

    # Check if all required input files exist
    for p in [name_path, desc_path, val_path]:
        if not os.path.exists(p):
            print(f"Error: File not found at {p}")
            return

    print(f"[-] Processing dataset: {dataset_key}")

    # 1. Load the target Subject IDs from the TXT file
    # Strip whitespace and store as a list
    with open(name_path, 'r') as f:
        train_subjects = [line.strip() for line in f.readlines()]
    
    # 2. Load the phenotype description file to get required variable names (columns)
    description_df = pd.read_csv(desc_path)
    if 'var_name' not in description_df.columns:
        print(f"Error: 'var_name' column not found in {args.description}")
        return
    target_vars = description_df['var_name'].tolist()

    # 3. Load the raw phenotype values
    # We only read the 'NAME' column and the variables listed in the description file
    cols_to_read = ['NAME'] + [v for v in target_vars]
    
    try:
        # Optimization: Use lambda to read only necessary columns to save memory
        phenotype_value_df = pd.read_csv(val_path, usecols=lambda x: x in cols_to_read)
    except Exception as e:
        print(f"Error reading {args.value}: {e}")
        return

    # 4. Filter rows based on Subject IDs
    # Convert 'NAME' to string to ensure matching works even if IDs look like numbers
    train_phenotype_df = phenotype_value_df[phenotype_value_df['NAME'].astype(str).isin(train_subjects)]

    # 5. Ensure column order: 'NAME' as the first column, followed by target variables
    # Filter only those variables that actually exist in the raw value file
    existing_vars = [v for v in target_vars if v in train_phenotype_df.columns]
    final_df = train_phenotype_df[['NAME'] + existing_vars]
    
    # 6. Save the filtered results to the output CSV
    final_df.to_csv(out_path, index=False)
    
    print(f"Success!")
    print(f"  > Output: {args.out}")
    print(f"  > Subjects found: {len(final_df)}")
    print(f"  > Variables included: {len(existing_vars)}")

if __name__ == "__main__":
    generate_train_phenotype_value()