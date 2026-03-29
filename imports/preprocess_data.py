import os
import warnings
import numpy as np
import scipy.io as sio
from nilearn import connectome

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Core Processing Functions
# -----------------------------------------------------------------------------

def get_ids(data_folder, num_subjects=None):
    """
    Reads the subject_IDs.txt file from the specified folder.

    Args:
        data_folder: Path to the folder containing 'subject_IDs.txt'.
        num_subjects: (Optional) Number of subjects to read (integer).

    Returns:
        subject_IDs: Numpy array of subject IDs.
    """
    file_path = os.path.join(data_folder, 'subject_IDs.txt')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[Error] The file {file_path} does not exist.")

    # ndmin=1 ensures result is always an array, even if only one subject exists
    subject_IDs = np.genfromtxt(file_path, dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


def get_timeseries(subject_list, atlas_name, data_folder, silence=False):
    """
    Reads time-series text files for a list of subjects.

    Args:
        subject_list : List of subject IDs (strings).
        atlas_name   : Name of the atlas used (e.g., 'Schaefer400').
        data_folder  : Root folder containing subject sub-folders.
        silence      : If True, suppresses print statements.

    Returns:
        time_series  : List of numpy arrays, each of shape (timepoints x regions).
    """
    timeseries = []

    # Expected suffix for the ROI file
    expected_suffix = '_rois_' + atlas_name + '.txt'

    for i in range(len(subject_list)):
        subject_id = subject_list[i]
        subject_folder = os.path.join(data_folder, subject_id)

        # Check if subject folder exists
        if not os.path.exists(subject_folder):
            if not silence:
                print(f"[Warning] Folder not found: {subject_folder}")
            continue

            # Find the specific ROI file
        all_files = os.listdir(subject_folder)
        ro_file = [f for f in all_files if f.endswith(expected_suffix)]

        if len(ro_file) == 0:
            if not silence:
                print(f"[Error] No ROI file found for subject: {subject_id} in {subject_folder}")
            continue

        fl = os.path.join(subject_folder, ro_file[0])

        if not silence:
            print("Reading timeseries file %s" % fl)

        try:
            timeseries.append(np.loadtxt(fl, skiprows=0))
        except Exception as e:
            print(f"[Error] Failed to read {fl}: {e}")

    return timeseries


def subject_connectivity(timeseries, subjects, atlas_name, kind, iter_no='', seed=1234,
                         n_subjects='', save=True, save_path=None):
    """
    Computes functional connectivity matrices from time-series data.

    Args:
        timeseries   : List of time-series arrays.
        subjects     : List of subject IDs.
        atlas_name   : Name of the parcellation atlas used.
        kind         : Type of connectivity (e.g., 'correlation', 'partial correlation', 'tangent').
        iter_no      : Iteration number (for cross-validation naming, optional).
        seed         : Random seed (optional).
        save         : Boolean, whether to save the matrix to disk.
        save_path    : Path to save the matrices (required if save=True).

    Returns:
        connectivity : The computed connectivity matrices.
    """

    if save and save_path is None:
        raise ValueError("Must provide 'save_path' when save=True")

    # Compute Connectivity
    if kind in ['TPE', 'TE', 'correlation', 'partial correlation']:
        if kind not in ['TPE', 'TE']:
            # Standard Correlation or Partial Correlation
            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            connectivity = conn_measure.fit_transform(timeseries)
        else:
            # Tangent Space Embedding
            if kind == 'TPE':
                conn_measure = connectome.ConnectivityMeasure(kind='correlation')
                conn_mat = conn_measure.fit_transform(timeseries)
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(conn_mat)
                connectivity = connectivity_fit.transform(conn_mat)
            else:
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(timeseries)
                connectivity = connectivity_fit.transform(timeseries)

    # Save to Disk
    if save:
        for i, subj_id in enumerate(subjects):
            # Ensure subject directory exists in the output path
            target_dir = os.path.join(save_path, subj_id)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # Construct filename
            if kind not in ['TPE', 'TE']:
                fname = f"{subj_id}_{atlas_name}_{kind.replace(' ', '_')}.mat"
            else:
                fname = f"{subj_id}_{atlas_name}_{kind.replace(' ', '_')}_{iter_no}_{seed}_{n_subjects}.mat"

            subject_file = os.path.join(target_dir, fname)

            # Save using scipy.io
            sio.savemat(subject_file, {'connectivity': connectivity[i]})

    return connectivity


def get_networks(subject_list, kind, data_folder, iter_no='', seed=1234, n_subjects='', atlas_name="aal",
                 variable='connectivity'):
    """
    Loads precomputed fMRI connectivity networks from disk.

    Args:
        subject_list : List of subject IDs.
        kind         : The kind of connectivity to load (e.g. 'correlation').
        data_folder  : Root folder where the .mat files are stored.
        atlas_name   : Name of the parcellation atlas used.
        variable     : Variable name inside the .mat file (default: 'connectivity').

    Returns:
        networks     : Stacked numpy array of connectivity networks (num_subjects x network_size).
    """

    all_networks = []

    # Normalize 'partial correlation' string to 'partial_correlation' for filenames
    kind_file_str = kind.replace(' ', '_')

    for subject in subject_list:
        fl = os.path.join(data_folder, subject,
                          f"{subject}_{atlas_name}_{kind_file_str}.mat")

        try:
            matrix = sio.loadmat(fl)[variable]
            all_networks.append(matrix)
        except FileNotFoundError:
            print(f"[Error] File not found: {fl}")
            continue

    if not all_networks:
        return np.array([])

    # Apply Fisher z-transform (arctanh) for standard correlations
    # Tangent space (TE/TPE) is already in a Euclidean space, so no transform needed.
    if kind in ['TE', 'TPE']:
        norm_networks = [mat for mat in all_networks]
    else:
        norm_networks = [np.arctanh(mat) for mat in all_networks]

    networks = np.stack(norm_networks)

    return networks