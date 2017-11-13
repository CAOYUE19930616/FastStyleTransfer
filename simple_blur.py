import keras
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from keras.preprocessing import image
import numpy as np
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Activation
from keras.losses import mean_squared_error
from keras.initializers import Constant
import matplotlib.pyplot as plt
#from neural_style_transfer import gram_matrix
import os

import tensorflow as tf
sess = tf.Session()
import keras.backend.tensorflow_backend as KTF
KTF.set_session(sess)

"2,5,9,13"
"""
style2 = K.function([vggnet.layers[0].input], [vggnet.layers[2].output])
style5 = K.function([vggnet.layers[0].input], [vggnet.layers[5].output])
style9 = K.function([vggnet.layers[0].input], [vggnet.layers[9].output])
style13 = K.function([vggnet.layers[0].input], [vggnet.layers[13].output])

mo9 = Model(inputs=vggnet.input, outputs=vggnet.layers[9].output)
"""

def getBlurNet(height, width):
    inputs = Input(shape=(height,width,3))
    #1
    x = Conv2D(3, (9,9), strides=1, padding='same',input_shape=(height,width,3),kernel_initializer=Constant(1.0/(3.0*9.0*9.0)),bias_initializer='zeros')(inputs)
    model = Model(inputs=inputs, outputs=x)

    return model

w=600
h=900

img = image.load_img("test_image/monkey.jpg", target_size=(w,h))
x = image.img_to_array(img)
x = np.expand_dims(x,0)
x = x/255

blur_net = getBlurNet(w,h)
blur_net.summary()

out = blur_net.predict([x])[0]
print(np.max(out))


plt.imshow(out)
plt.show()

"""
def feature_loss(y_true, y_pred):
    yt = mo9.predict(y_true)
    yp = mo9.predict(y_pred)
    #return K.sum(K.square(y_true - y_pred),axis=[1,2,3])
    return K.sum(K.square(y_true-y_pred))
    #return K.mean(style9([y_true])-style9([y_pred]),axis=[1,2,3])

y = feature_loss(x,x)
print(y)
print(y.shape)


def style_loss(y_true, y_pred, n):
    return K.sum(gram_matrix(style(n)(y_true)) - gram_matrix(style(n)(y_pred)))

transform_net = getTransformNet(224,224)
#transform_net.summary()
transform_net.compile(optimizer='adam', loss=[feature_loss])
"""
f = open("models/blur1.txt", "w")

[f.write(n.name+"\n") for n in sess.graph.as_graph_def().node]
saver = tf.train.Saver()
saver.save(sess, "models/blur1.ckpt")
tf.train.write_graph(sess.graph.as_graph_def(), "models", "blur1.pb")
