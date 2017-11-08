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
import matplotlib.pyplot as plt
#from neural_style_transfer import gram_matrix



"2,5,9,13"
"""
style2 = K.function([vggnet.layers[0].input], [vggnet.layers[2].output])
style5 = K.function([vggnet.layers[0].input], [vggnet.layers[5].output])
style9 = K.function([vggnet.layers[0].input], [vggnet.layers[9].output])
style13 = K.function([vggnet.layers[0].input], [vggnet.layers[13].output])

mo9 = Model(inputs=vggnet.input, outputs=vggnet.layers[9].output)
"""

def getTransformNet(height, width):
    inputs = Input(shape=(height,width,3))
    #1
    x = Conv2D(32, (9,9), strides=1, padding='same',input_shape=(height,width,3))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #2
    x = Conv2D(64, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #3
    x = Conv2D(128, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(0,5):
        x = Conv2D(128, (3,3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3,3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2DTranspose(64,(3,3),strides=2,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32,(3,3),strides=2,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(3, (9,9), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #model = Model(inputs=inputs, outputs=x)
    vggnet = VGG16(include_top=False, input_shape=(height,width,3))

    for layer in vggnet.layers[1:]:
        x = layer(x)

    model = Model(inputs=inputs, outputs=x)

    return model

def getWholeNet(height, width):
    vggnet = VGG16(include_top=False, input_shape=(height,width,3))
    tn = getTransformNet(height,width)
    for layer in vggnet.layers[1:]:
        tn = layer(tn)
    x = Model(inputs=tn.input, outputs=tn.output)
    return x



"""
img = image.load_img("test_image/monkey.jpg", target_size=(224,224))
x = image.img_to_array(img)
x = np.stack((x,x))
"""

whole_net = getTransformNet(256,256)
whole_net.summary()

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