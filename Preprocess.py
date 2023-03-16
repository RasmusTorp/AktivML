import os
from skimage import io, color, morphology
import matplotlib.pyplot as plt
import numpy as np 
import cv2

# Windows
folder_path = os.getcwd() + "\data\shipsnet\shipsnet"

# Linux
folder_path = "ActiveProjectBO/data/shipsnet/shipsnet"


img_list = []
label_list = []
img_list_gray = []
img_list_rgb = []

def process(img):
    # Border median
    img_border = np.concatenate((img[:,0], img[:,-1], img[0,:], img[-1,:]))
    bm = np.median(img_border, axis=0)
    
    # Color thresholding
    dif = 10
    thr = cv2.inRange(img[:,:,0], bm[0]-dif, bm[0]+dif)
    thg = cv2.inRange(img[:,:,1], bm[1]-dif, bm[1]+dif)
    thb = cv2.inRange(img[:,:,2], bm[2]-dif, bm[2]+dif)
    thcomb = thr & thg & thb
    mask = np.logical_not(thcomb)
    
    # Remove small objects
    mask_nonoise = morphology.remove_small_objects(mask, 10)
    img_rgb = cv2.bitwise_and(img,img, mask= np.uint8(mask_nonoise))

    # RGB to Gray
    img_gray = color.rgb2gray(img_rgb)
    #img_gray = np.reshape(img_gray, (3,3,1))
    img_gray= np.expand_dims(img_gray, axis=2)
    return img_rgb, img_gray

def simple_process(img):
    return color.rgb2gray(img)


# Main loop
for i, filename in enumerate(os.listdir(folder_path)):
    file = os.path.join(folder_path,filename)
    if os.path.isfile(file):
        img = io.imread(file)
        img_list.append(img)
        label_list.append(int(filename[0]))
        img_gray = simple_process(img)
        img_list_rgb.append(img)
        img_list_gray.append(img_gray)
        print(i)

# io.imshow_collection([img_list[15], img_list_gray[15], img_list_rgb[15]])
#io.imshow_collection([img_list[15], img_list_gray[15]], cmap="gray")
io.imshow_collection(img_list[12:21])
plt.show()
# print()

# np.save("image_data.npy",np.array(img_list))
np.save("labels.npy",np.array(label_list))
np.save("image_data_gray.npy", np.array(img_list_gray))
np.save("image_data_rgb.npy", np.array(img_list_rgb))

