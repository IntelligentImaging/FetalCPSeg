import os
import glob

import nibabel as nib
import numpy as np
import random
import torch
from torch.utils.data import Dataset


def get_list(dir_path):
    """
    This function is to get the data path list of the dataset.
    Note that:
    the data should be placed like the following layout

    data_folder
        -Subject-1
            -image.nii.gz
            -label.nii.gz
        -Subject-2
        -Subject-3
        ...

    Besides, all of the image data should be rescaled to [0, 1].
    """

    print("Reading Data...")

    dict_list = []

    path_list = glob.glob(os.path.join(dir_path, '*'))

    path_list.sort()

    image_name = 'image.nii.gz'  # name of image data
    label_name = 'label.nii.gz'  # name of label data.

    for path in path_list:
        dict_list.append(
            {
                'image_path': os.path.join(path, image_name),
                'label_path': os.path.join(path, label_name),
                'save_path': path
            }
        )

    print("Finished! Data NumA:{}".format(len(dict_list)))

    return dict_list
