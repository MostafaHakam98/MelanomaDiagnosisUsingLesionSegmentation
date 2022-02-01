import cv2

from mpl_toolkits.axes_grid1 import ImageGrid
from os.path import isfile, join
from random import shuffle
from os import listdir


def generate_table(root_files, root_labels, file_paths, label_paths, inference, n_rows):
    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n_rows, 3),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    grid.axes_all[0].set_title('Original Image')
    grid.axes_all[1].set_title('Prediction')
    grid.axes_all[2].set_title('Ground Truth')
    for i in range(n_rows):
        img_path = join(root_files, file_paths[i])
        index_label = label_paths.index(file_paths[i][0:-4] + '_segmentation.png')
        label_path = join(root_labels, label_paths[index_label])
        _, pred = inference.predict(img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resize = cv2.resize(image, (224, 224))
        label = cv2.imread(label_path, 0)
        label_resize = cv2.resize(label, (224, 224))

        ax = grid[(i * 3)]
        ax.imshow(image_resize)
        ax = grid[(i * 3) + 1]
        ax.imshow(pred, cmap='gray')
        ax = grid[(i * 3) + 2]
        ax.imshow(label_resize, cmap='gray')
