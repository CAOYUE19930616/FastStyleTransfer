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
style_dir = './VanGoghStarryNight1000'
content_dir = '../COCO/test2015_256x256'

vgg_net = getVggNet(height, width)

#[index, feature1, feature2,feature3,feature4]

# make x (input)
content_images = []
content_names = os.listdir(content_dir)
for name in content_names[0:336]:
    img = image.load_img(content_dir+'/'+name, target_size=(height,width))
    x = image.img_to_array(img)
    content_images.append(x)
    time.sleep(0.01)
content_images_array = np.stack(content_images)
content_features = vgg_net.predict(content_images_array, batch_size=1)
time.sleep(5.0)

# make y (answer)
style_images = []
style_names = os.listdir(style_dir)
for name in style_names:
    img = image.load_img(style_dir+'/'+name, target_size=(height,width))
    x = image.img_to_array(img)
    style_images.append(x)
    time.sleep(0.01)

style_images_array = np.stack(style_images)
style_features = vgg_net.predict(style_images_array, batch_size=1)
time.sleep(5.0)

"""
for i in range(0,len(content_images)):
    sf = random.choice(style_features)
    feat = []
    feat.append(content_images[i])
    feat.extend(sf)
    feat.append(content_features[2])
    y.append(feat)
"""

model = getWholeNet(height, width)
model.compile(optimizer='adam', 
              loss=[TVLoss, styleLoss1, styleLoss3, styleLoss3, styleLoss4, featureLoss],
              loss_weights=[1, 1, 1, 1, 1, 1e-5])

model.fit(content_images_array, [content_images_array, style_features[0], style_features[1], style_features[2], style_features[3], content_features[2]], 
          batch_size=1, epochs=100)
