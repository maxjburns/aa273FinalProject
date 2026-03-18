import numpy as np

def load_npz(file_path):
    """
    Load an .npz file and return its contents as a dictionary.

    Parameters
    ----------
    file_path : str
        Path to the .npz file.

    Returns
    -------
    data_dict : dict
        Dictionary where keys are the variable names in the .npz file
        and values are the corresponding numpy arrays.
    """
    npz_file = np.load(file_path, allow_pickle=True)
    data_dict = {key: npz_file[key] for key in npz_file.files}
    npz_file.close()
    return data_dict