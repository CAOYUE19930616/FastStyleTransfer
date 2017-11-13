import numpy as np
import scipy.misc as misc
import os

input_dir = "./style_image"
filename = "VanGoghStarryNight1000.jpg"
output_dir, ext = os.path.splitext(filename)
os.mkdir(output_dir)
strides = 50

patch_width = 256
patch_height = 256

img = misc.imread(input_dir+'/'+filename)

height = img.shape[0]
width = img.shape[1]

k=0
h = 0
w = 0

while h+patch_height<height:
    while w+patch_width<width:
        patch = img[h:h+patch_height,w:w+patch_width,:]
        misc.imsave(output_dir + '/' + str(k) + ext, patch)
        k+=1
        w+=strides
    w = 0
    h += strides
