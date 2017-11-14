import keras
import numpy as np
from keras.preprocessing import image
from functional_model import *
import os
import tensorflow as tf
from keras.backend import tensorflow_backend
import time
import random

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

height = 256
width = 256
num_data = 500
style_dir = './VanGoghStarryNight1000'
content_dir = '../COCO/test2015_256x256'
output_dir = './VanGoghStarryNight1000_data'
try:
    os.mkdir(output_dir)
except:
    pass

vgg_net = getVggNet(height, width)

# make x (input)
content_images = []
content_names = os.listdir(content_dir)
for name in content_names[0:num_data]:
    img = image.load_img(content_dir+'/'+name, target_size=(height,width))
    x = image.img_to_array(img)
    content_images.append(x)
content_images_array = np.stack(content_images)
content_features = vgg_net.predict(content_images_array, batch_size=1)
time.sleep(5.0)
np.save(output_dir + "/cf.npy", content_features[2])
np.save(output_dir + "/images.npy", content_images_array)
"""
# make y (answer)
style_images = []
style_names = os.listdir(style_dir)
for name in style_names:
    img = image.load_img(style_dir+'/'+name, target_size=(height,width))
    x = image.img_to_array(img)
    style_images.append(x)

style_images_extend = []
for i in range(0,num_data):
    style_images_extend.append(random.choice(style_images))

style_images_array = np.stack(style_images_extend)
style_features = vgg_net.predict(style_images_array, batch_size=1)
time.sleep(5.0)
np.save(output_dir + "/sf1.npy", style_features[0])
np.save(output_dir + "/sf2.npy", style_features[1])
np.save(output_dir + "/sf3.npy", style_features[2])
np.save(output_dir + "/sf4.npy", style_features[3])
"""
