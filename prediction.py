import keras
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from functional_model import *
import os
import tensorflow as tf
from keras.backend import tensorflow_backend
import time
import random
import scipy.misc as misc

height = 256
width = 256

test_dir = '../COCO/test2015_256x256'
num_data=10
test_images = []
test_names = os.listdir(test_dir)
for name in test_names[0:num_data]:
    img = image.load_img(test_dir+'/'+name, target_size=(height,width))
    x = image.img_to_array(img)
    test_images.append(x)
test_images_array = np.stack(test_images)

model = keras.models.load_model('./tmp.h5',custom_objects={'TVLoss':TVLoss, 'styleLoss1':styleLoss1, 'styleLoss2':styleLoss2, 'styleLoss3':styleLoss3, 'styleLoss4':styleLoss4, 'featureLoss':featureLoss})
result = model.predict(test_images_array)[0]

for i in range(0,num_data):
    misc.imsave('./out/'+str(i)+'.jpg', result[i])
    i+=1

