import os
import random

import matplotlib.pyplot as plt
import pandas
from skimage.draw import polygon
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import json

TYPE_NUCLEI_DICT = {
    1: "Opal_480", #podocytes
    2: "Opal_520", #PECs
    3: "Opal_570", #endothelial
    4: "Opal_620", #mesangial
    5: "Opal_690", #immune
    5: "Outside",
    5: "Unclassified"
}

if __name__ == "__main__":
    # WSIs_path = "/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/fold1/WSIs/"
    # GTs_geojson_path = "/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/fold1/GTs_geojson/"
    # ROIs_geojson_path = "/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/fold1/ROIs_geojson/"
    # images_path = "/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/fold1/images/"
    # labels_path = "/scratch/nmoreau/CellViT_2025/kidney_data_256_40x/fold1/labels/"

    folder_path = "/scratch/nmoreau/Test_data/"

    # folder_path = "/Users/nmoreau/Documents/Data/Kidney/new_organization/processed_data/cellvit/kidney_data_256_20x/fold2/"

    WSIs_path = folder_path + "/WSIs/"
    GTs_geojson_path = folder_path + "/GTs_geojson/"
    ROIs_geojson_path = folder_path + "/ROIs_geojson/"
    images_path = folder_path + "/images/"
    labels_path = folder_path + "/labels/"
    TYPE_NUCLEI_DICT_inv = {TYPE_NUCLEI_DICT[k]: k for k in TYPE_NUCLEI_DICT.keys()}
    patch_size = (256, 256)
    cells_count_json = {
        "images": [],
        "Opal_480": [],
        "Opal_520": [],
        "Opal_570": [],
        "Opal_620": [],
        "Unclassified": []
        # "Opal_690": [],
        # "Outside": [],
        # "Unclassified": []
    }
    types_json = {
        "img": [],
        "type": []
    }
    for image_name in os.listdir(WSIs_path):
        if not image_name.startswith("."):
            image_name = image_name[:-8]
            print(image_name)

            with open(GTs_geojson_path + image_name + ".geojson", 'r') as f:
                gson_cells_gt = json.load(f)
            with open(ROIs_geojson_path + image_name + ".geojson", 'r') as f:
                gson_rois_gt = json.load(f)
            rois_list = gson_rois_gt["features"]
            cells_gt_list = gson_cells_gt["features"]

            WSI_pil = Image.open(WSIs_path + image_name + "_PAS.png")
            WSI_array = np.array(WSI_pil)

            for roi in rois_list:
                roi_id = roi["id"]

                x_roi_list = [coord[0] for coord in roi["geometry"]["coordinates"][0]]
                y_roi_list = [coord[1] for coord in roi["geometry"]["coordinates"][0]]
                xmax = max(x_roi_list)
                xmin = min(x_roi_list)
                ymax = max(y_roi_list)
                ymin = min(y_roi_list)

                WSI_roi = WSI_array[ymin:ymax, xmin:xmax, :3]
                GT_type_map = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint16)
                GT_inst_map = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint16)
                i = 0
                for cell in cells_gt_list:
                    if cell["properties"]["objectType"] == "cell":
                        id_cell = cell["id"]
                        list_coord_cell = cell["geometry"]["coordinates"][0]
                        properties = cell["properties"]
                        if "classification" in properties.keys():
                            name = properties["classification"]["name"]
                        else:
                            name = "Unclassified"
                        if name == "Opal_690":
                            name = "Unclassified"
                        x1 = list_coord_cell[0][0]
                        y1 = list_coord_cell[0][1]
                        if xmin < x1 < xmax and ymin < y1 < ymax:
                            i += 1
                            new_list_coord_cell = []
                            new_list_coord_nuclear = []
                            for coord in list_coord_cell:
                                new_coord_cell = [coord[0] - xmin, coord[1] - ymin]
                                new_list_coord_cell.append(new_coord_cell)
                            poly = np.array(new_list_coord_cell[:-1])
                            rr, cc = polygon(poly[:, 0], poly[:, 1], (GT_inst_map.shape[1], GT_inst_map.shape[0]))
                            GT_inst_map[cc, rr] = i
                            GT_type_map[cc, rr] = TYPE_NUCLEI_DICT_inv[name]
                WSI_roi_pil = Image.fromarray(WSI_roi)
                outdict = {"inst_map": GT_inst_map, "type_map": GT_type_map}
                WSI_roi_pil.save(
                    images_path + image_name + "_" + str(roi_id) + ".png")
                np.save(labels_path + image_name + "_" + str(roi_id) + ".npy",
                    outdict)
    #             path_number = 0
    #             for x in range(0, WSI_roi.shape[0], patch_size[0]):
    #                 for y in range(0, WSI_roi.shape[1], patch_size[1]):
    #                     path_number += 1
    #                     WSI_patch = WSI_roi[x:x + patch_size[0], y:y + patch_size[1]]
    #                     GT_inst_map_patch = GT_inst_map[x:x + patch_size[0], y:y + patch_size[1]]
    #                     GT_type_map_patch = GT_type_map[x:x + patch_size[0], y:y + patch_size[1]]
    #                     # rand = random.randrange(5)
    #                     rand = 0
    #                     if GT_inst_map_patch.shape == patch_size and rand == 0:
    #                         # print(WSI_patch.shape)
    #                         WSI_patch_pil = Image.fromarray(WSI_patch)
    #                         outdict = {"inst_map": GT_inst_map_patch, "type_map": GT_type_map_patch}
    #
    #                         types_json["img"].append(image_name + "_" + str(roi_id) + "_" + str(path_number) + ".png")
    #                         types_json["type"].append("kidney")
    #
    #                         cells_count_json["images"].append(image_name + "_" + str(roi_id) + "_" + str(path_number) + ".png")
    #                         cells_count_json["Opal_480"].append(0)
    #                         cells_count_json["Opal_520"].append(0)
    #                         cells_count_json["Opal_570"].append(0)
    #                         cells_count_json["Opal_620"].append(0)
    #                         cells_count_json["Unclassified"].append(0)
    #                         # cells_count_json["Opal_690"].append(0)
    #                         # cells_count_json["Outside"].append(0)
    #                         # cells_count_json["Unclassified"].append(0)
    #
    #                         for cell_inst in np.unique(GT_inst_map_patch):
    #                             if cell_inst != 0:
    #                                 cell_type = np.argmax(np.bincount(GT_type_map_patch[GT_inst_map_patch == cell_inst]))
    #                                 if cell_type == 5:
    #                                     cells_count_json["Unclassified"][-1] = cells_count_json["Unclassified"][-1] + 1
    #                                 else:
    #                                     cell_type = TYPE_NUCLEI_DICT[cell_type]
    #                                     cells_count_json[cell_type][-1]=cells_count_json[cell_type][-1]+1
    #                         nb_cell = np.unique(GT_inst_map_patch).shape[0]
    #                         if False:
    #                         # if cells_count_json["Unclassified"][-1] > nb_cell/4:
    #                             cells_count_json["images"].pop()
    #                             cells_count_json["Opal_480"].pop()
    #                             cells_count_json["Opal_520"].pop()
    #                             cells_count_json["Opal_570"].pop()
    #                             cells_count_json["Opal_620"].pop()
    #                             cells_count_json["Unclassified"].pop()
    #                         else :
    #                             WSI_patch_pil.save(
    #                                 images_path + image_name + "_" + str(roi_id) + "_" + str(path_number) + ".png")
    #                             np.save(labels_path + image_name + "_" + str(roi_id) + "_" + str(path_number) + ".npy",
    #                                 outdict)
    #                         # plt.imshow(WSI_patch)
    #                         # plt.show()
    #                         # plt.imshow(GT_inst_map_patch)
    #                         # plt.show()
    #                         # plt.imshow(GT_type_map_patch)
    #                         # plt.show()
    #             # plt.imshow(WSI_roi)
    #             # plt.show()
    #             # plt.imshow(GT_inst_map)
    #             # plt.show()
    #             # plt.imshow(GT_type_map)
    #             # plt.show()
    #
    # cell_count = pandas.DataFrame(cells_count_json,
    #                               columns=["images", "Opal_480", "Opal_520",
    #                                        "Opal_570",
    #                                        "Opal_620", "Unclassified"])
    # cell_count.to_csv(folder_path + "/cell_count.csv", ';')
    #
    # types = pandas.DataFrame(types_json,
    #                               columns=["img", "type"])
    # types.to_csv(folder_path + "/types.csv", ';')