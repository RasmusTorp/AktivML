import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
from skimage import io

LOADPATH = "top10_ranked_images_50.npy"
LABELPATH = "top10_ranked_images_labels_50.npy"
imgs = np.load(LOADPATH)
labels = np.load(LABELPATH)
print(labels)


# plt.imshow(first[0])
# io.imshow_collection(imgs[:7])
# plt.show()


fig = plt.figure(figsize=(5., 5.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                 axes_pad=(0.1, 0.3),  # pad between axes in inch.
                 )

# for ax, im in zip(grid, [im1, im2, im3, im4]):
for ax, (i, im) in zip(grid, enumerate(imgs[:4])):
    # Iterating over the grid returns the Axes.
    ax.set_title(f"Label: {labels[i]}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im)

plt.savefig("top_rank_imgs.png")

print()


