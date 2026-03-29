"""
Script Name: 10-preprocessing_Image_data.py
Description:
    This script serves as a preprocessing pipeline for fMRI data. It automates the
    creation of directory structures, calculation of functional connectivity matrices
    (Pearson and Partial Correlation) from time-series data, and packaging of
    the results into individual HDF5 (.h5) files for deep learning models.

Inputs:
    1. Subject ID File: A text file (Subject_ID.txt) containing a list of subject identifiers.
    2. ROI Time-series Data: Text files containing time-series data for each subject.
       - Format: .txt files
       - Shape: (Timepoints x ROIs), e.g., (1200 x 400).
       - Location: Defined by 'input_vol_root'.

Outputs:
    1. Intermediate .mat files: Correlation and Partial Correlation matrices saved in subject-specific folders.
    2. Final .h5 files: Single HDF5 file per subject containing the processed matrices and a placeholder label.
       - Location: [output_root]/raw/subject_id.h5

Data Distribution & Format (.h5 structure):
    Each generated .h5 file contains a dictionary with the following keys:
    - 'corr':  Pearson Correlation Matrix.
               Shape: (N_ROIs, N_ROIs), e.g., (400, 400).
               Range: [-1, 1]. Float.
    - 'pcorr': Partial Correlation Matrix.
               Shape: (N_ROIs, N_ROIs), e.g., (400, 400).
               Range: Roughly [-1, 1], varies based on precision matrix. Float.
    - 'label': Classification label.
               Format: Integer (Currently hardcoded to 0 for unsupervised/pre-training tasks).

Usage:
    python 10-preprocessing_Image_data.py --dataset ABCD
    python 10-preprocessing_Image_data.py --dataset HCP_A
"""

import argparse
import os
import sys
import shutil
import deepdish as dd
import warnings

# Import custom data reading module
# Ensure the 'imports' folder is in the current directory or PYTHONPATH
from imports import preprocess_data as Reader

warnings.filterwarnings("ignore")


def main():
    # ---------------------------------------------------------
    # 1. Parameter Settings
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Pipeline: Create folders, fetch data, compute connectivity, and save to H5.')
    parser.add_argument('--dataset', type=str, default='ABCD',
                        choices=['ABCD', 'HCP_A', 'HCP_D', 'HCP_YA'],
                        help='Choose dataset: ABCD, HCP_A, HCP_D, or HCP_YA')
    parser.add_argument('--atlas', default='Schaefer400',
                        help='Brain parcellation atlas. Options: ho, cc200, cc400, Schaefer400. Default: Schaefer400.')
    parser.add_argument('--nclass', default=2, type=int,
                        help='Number of classes for classification label. Default: 2')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialization.')

    args = parser.parse_args()

    print(f"[-] Starting Preprocessing for Dataset: {args.dataset}")
    print(f"[-] Atlas: {args.atlas}")

    # ---------------------------------------------------------
    # 2. Path and Configuration Definitions
    # ---------------------------------------------------------
    # Base input root directory
    base_input_root = '/data/hzb/project/OBOP_data'
    # Current working directory (for relative path creation)
    current_cwd = os.getcwd()

    input_id_file = ""
    input_vol_root = ""
    output_folder_root = ""

    if args.dataset == 'ABCD':
        # --- ABCD Configuration ---
        input_id_file = os.path.join(base_input_root, 'ABCD', 'ABCD_subject_id', 'Subject_ID.txt')
        input_vol_root = os.path.join(base_input_root, 'ABCD', 'ABCD_vol_mean')
        output_folder_root = os.path.join(current_cwd, 'data', 'ABCD_pcp', 'Schaefer', 'filt_noglobal')

    elif args.dataset == 'HCP_A':
        # --- HCP-A Configuration ---
        input_id_file = os.path.join(base_input_root, 'HCP_A', 'HCP_A_subject_id', 'Subject_ID.txt')
        input_vol_root = os.path.join(base_input_root, 'HCP_A', 'HCP_A_vol_mean')
        output_folder_root = os.path.join(current_cwd, 'data', 'HCP_A_pcp', 'Schaefer', 'filt_noglobal')

    elif args.dataset == 'HCP_D':
        # --- HCP-D Configuration ---
        input_id_file = os.path.join(base_input_root, 'HCP_D', 'HCP_D_subject_id', 'Subject_ID.txt')
        input_vol_root = os.path.join(base_input_root, 'HCP_D', 'HCP_D_vol_mean')
        output_folder_root = os.path.join(current_cwd, 'data', 'HCP_D_pcp', 'Schaefer', 'filt_noglobal')

    elif args.dataset == 'HCP_YA':
        # --- HCP-YA Configuration ---
        input_id_file = os.path.join(base_input_root, 'HCP_YA', 'HCP_YA_subject_id', 'Subject_ID.txt')
        input_vol_root = os.path.join(base_input_root, 'HCP_YA', 'HCP_YA_vol_mean')
        output_folder_root = os.path.join(current_cwd, 'data', 'HCP_YA_pcp', 'Schaefer', 'filt_noglobal')

    # Check if input file exists
    if not os.path.exists(input_id_file):
        print(f"[Error] Subject ID file not found at: {input_id_file}")
        sys.exit(1)

    # Create output root directory
    if not os.path.exists(output_folder_root):
        os.makedirs(output_folder_root)
        print(f"[-] Created output root: {output_folder_root}")

    # Copy Subject ID file to output directory (for record keeping)
    shutil.copyfile(input_id_file, os.path.join(output_folder_root, 'subject_IDs.txt'))

    # ---------------------------------------------------------
    # 3. Phase 1: Create Directory Structure
    # ---------------------------------------------------------
    print("\n--- Phase 1: Creating Directory Structure ---")
    with open(input_id_file, 'r', encoding='utf-8') as file:
        raw_subject_lines = file.readlines()

    valid_subjects_list = []
    for line in raw_subject_lines:
        folder_name = line.strip()
        if folder_name:
            valid_subjects_list.append(folder_name)
            target_path = os.path.join(output_folder_root, folder_name)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

    print(f"[-] Directories ensured for {len(valid_subjects_list)} subjects.")

    # ---------------------------------------------------------
    # 4. Phase 2: Compute Connectivity Matrices
    # ---------------------------------------------------------
    print("\n--- Phase 2: Computing Connectivity Matrices ---")

    # Get Subject IDs using the Reader module (reads from the output folder we just populated)
    subject_IDs = Reader.get_ids(output_folder_root)

    # Ensure ID list is a standard python list
    if hasattr(subject_IDs, 'tolist'):
        subject_IDs = subject_IDs.tolist()

    # Get Time Series data
    print("[-] Fetching time series...")
    time_series = Reader.get_timeseries(subject_IDs, args.atlas, data_folder=input_vol_root)

    # Compute and save matrices (Intermediate step)
    print(f"[-] Computing Correlation matrices...")
    Reader.subject_connectivity(time_series, subject_IDs, args.atlas, 'correlation', save_path=output_folder_root)
    print(f"[-] Computing Partial Correlation matrices...")
    Reader.subject_connectivity(time_series, subject_IDs, args.atlas, 'partial correlation', save_path=output_folder_root)

    # ---------------------------------------------------------
    # 5. Phase 3: Packaging Data to H5
    # ---------------------------------------------------------
    print("\n--- Phase 3: Packaging Data to H5 ---")

    # Retrieve the computed network features from disk
    fea_corr = Reader.get_networks(subject_IDs, iter_no='', kind='correlation', atlas_name=args.atlas, data_folder=output_folder_root)
    fea_pcorr = Reader.get_networks(subject_IDs, iter_no='', kind='partial correlation', atlas_name=args.atlas, data_folder=output_folder_root)

    # Prepare directory for raw H5 data
    raw_h5_dir = os.path.join(output_folder_root, 'raw')
    if not os.path.exists(raw_h5_dir):
        os.makedirs(raw_h5_dir)

    # Loop through subjects and save individual H5 files
    print(f"[-] Saving individual .h5 files to {raw_h5_dir} ...")
    count_saved = 0
    for i, subject in enumerate(subject_IDs):
        save_path = os.path.join(raw_h5_dir, subject + '.h5')

        # Create data dictionary
        # 'label' is hardcoded to 0 as per current requirement (e.g., for unsupervised learning or placeholder)
        data_dict = {
            'corr': fea_corr[i],
            'pcorr': fea_pcorr[i],
            'label': 0
        }

        dd.io.save(save_path, data_dict)
        count_saved += 1

    print(f"[Success] Completed. Processed {count_saved} subjects for dataset {args.dataset}.")


if __name__ == '__main__':
    main()