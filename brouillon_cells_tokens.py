import numpy as np
import torch
import umap
from matplotlib import pyplot as plt
import json as geojson

if __name__ == "__main__":
    file_path = "/Users/nmoreau/Documents/Data/Kidney/new_organization/processed_data/cellvit/json_last/cell_detection_CellViT-256-x20_kidney_seg_forth/cells.pt"
    file_cell_vit = "/Users/nmoreau/Documents/Data/Kidney/new_organization/processed_data/cellvit/json_last/cell_detection_CellViT-256-x20_kidney_seg_forth/new_O_hNiere_S1.geojson"

    with open(file_cell_vit, 'r') as f:
        gson_cells_auto = geojson.load(f)

    cells_auto_list = gson_cells_auto["features"]

    graph_data = torch.load(file_path)
    contours_tensor = graph_data.contours
    contours_array = [tensor.numpy().astype("int32") for tensor in contours_tensor]
    embeddings_tensor = graph_data.x
    embeddings_array = embeddings_tensor.numpy()
    # print("ok")

    list_classes = []
    list_embeddings = []

    # for contour in contours_array:
    #     pass
    contours_array_auto = []
    class_auto = []
    # for c in range(len(contours_array)):
    #     if contours_array[c] in
    for cell in cells_auto_list:
        if cell["geometry"]["type"] == "Polygon":
            x_list = [coord[0] for coord in cell["geometry"]["coordinates"][0]]
            y_list = [coord[1] for coord in cell["geometry"]["coordinates"][0]]
            center = [sum(x_list) / len(x_list), sum(y_list) / len(y_list)]
            contour = np.array(cell["geometry"]["coordinates"][0])[:2]
            contours_array_auto.append(center)
            properties = cell["properties"]
            if "classification" in properties.keys():
                name = properties["classification"]["name"]
                if name == "Opal_480":
                    class_auto.append(1)
                elif name == "Opal_520":
                    class_auto.append(2)
                elif name == "Opal_570":
                    class_auto.append(3)
                elif name == "Opal_620":
                    class_auto.append(4)
                elif name == "Opal_690":
                    class_auto.append(5)
                elif name == "Unclassified":
                    class_auto.append(6)
                else:
                    class_auto.append(0)
            else:
                class_auto.append(0)
    print(len(contours_array_auto))
    contours_array_auto = np.asarray(contours_array_auto)
    for c in range(0, len(contours_array)):
        x_list = [coord[0] for coord in contours_array[c]]
        y_list = [coord[1] for coord in contours_array[c]]
        center = np.array((sum(x_list) / len(x_list), sum(y_list) / len(y_list)))
        dist_2 = np.sum((contours_array_auto - center)**2, axis=1)
        i = np.argmin(dist_2)
        if dist_2[i] < 1:
            # if True:
            if class_auto[i] != 0 and class_auto[i] != 5 and class_auto[i] != 6:
                list_classes.append(class_auto[i])
                list_embeddings.append(embeddings_array[c])
            # print(center)
            # print(contours_array_auto[i])
            # print(dist_2[i])
            # print("ok")

        # bo = [i for i in range(len(contours_array_auto)) if (contours_array_auto[i] == center)]
        # if len(bo) > 0:
        #     i = bo[0]
        #     # if True :
        #     if class_auto[i] != 0:
        #         list_classes.append(class_auto[i])
        #         list_embeddings.append(embeddings_array[c])
        if c % 10000 == 0 and c > 0:
            print(c, len(contours_array))
            print(len(list_embeddings))
            # break
        # if (contours_array[c] in contours_array_auto).all():
        #     print("ok")
        # index = [i for i in range(len(contours_array)) if (contours_array[i] == contour).all()]
        # list_embeddings.append(embeddings_array[index])

    # np.random.seed(42)
    # data = np.random.rand(800, 4)
    print(len(list_embeddings))
    reducer = umap.UMAP(n_components=3)
    u = reducer.fit_transform(list_embeddings)

    plt.scatter(u[:, 0], u[:, 1], c=list_classes, cmap="Spectral")
    plt.colorbar(boundaries=np.arange(8) - 0.5).set_ticks(np.arange(7))
    plt.title('UMAP embedding of random colours')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=list_classes, cmap="Spectral")
    # ax.colorbar(boundaries=np.arange(8) - 0.5).set_ticks(np.arange(7))
    plt.title('UMAP embedding of random colours')
    plt.show()