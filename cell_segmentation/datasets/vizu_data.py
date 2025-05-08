import os
import random

import matplotlib.pyplot as plt
import pandas
from skimage.draw import polygon
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import json



if __name__ == "__main__":

    folder_path = "/scratch/nmoreau/Test_data_tris/fold2/"

    # folder_path = "/Users/nmoreau/Documents/Data/Kidney/new_organization/processed_data/cellvit/kidney_data_256_20x/fold2/"

    WSIs_path = folder_path + "/WSIs/"
    GTs_geojson_path = folder_path + "/GTs_geojson/"
    ROIs_geojson_path = folder_path + "/ROIs_geojson/"
    images_path = folder_path + "/images/"
    labels_path = folder_path + "/labels/"

    for patch_name in os.listdir(images_path):
        if not patch_name.startswith("."):
            print(patch_name)
            patch_pil = Image.open(images_path + patch_name )
            patch_array = np.array(patch_pil)
            print(patch_array.shape)