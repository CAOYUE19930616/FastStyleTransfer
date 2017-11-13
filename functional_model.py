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

def getWholeNet(height, width):
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
    out0 = Activation('relu')(x)    #output_image

    #model = Model(inputs=inputs, outputs=x)
    vggnet = VGG16(include_top=False, input_shape=(height,width,3))

    for layer in vggnet.layers:
        layer.trainable = False

    x = vggnet.layers[1](out0)         #block1_conv1
    out1 = vggnet.layers[2](x)      #block1_conv2
    x = vggnet.layers[3](out1)      #block1_pool
    
    x = vggnet.layers[4](x)         #block2_conv1
    out2 = vggnet.layers[5](x)      #block2_conv2
    x = vggnet.layers[6](out2)      #block2_pool

    x = vggnet.layers[7](x)         #block3_conv1
    x = vggnet.layers[8](x)         #block3_conv2
    out3 = vggnet.layers[9](x)      #block3_conv3
    x = vggnet.layers[10](out3)        #block3_pool

    x = vggnet.layers[11](x)         #block3_conv1
    x = vggnet.layers[12](x)         #block3_conv2
    out4 = vggnet.layers[13](x)      #block3_conv3

    """
    for layer in vggnet.layers[1:]:
        layer.trainable = False
        x = layer(x)
    """

    model = Model(inputs=inputs, outputs=[out0, out1, out2, out3, out4, out3])

    return model

def getVggNet(height, width):
    vggnet = VGG16(include_top=False, input_shape=(height,width,3))
    inputs = Input(shape=(height,width,3))

    for layer in vggnet.layers:
        layer.trainable = False

    x = vggnet.layers[1](inputs)         #block1_conv1
    out1 = vggnet.layers[2](x)      #block1_conv2
    x = vggnet.layers[3](out1)      #block1_pool
    
    x = vggnet.layers[4](x)         #block2_conv1
    out2 = vggnet.layers[5](x)      #block2_conv2
    x = vggnet.layers[6](out2)      #block2_pool

    x = vggnet.layers[7](x)         #block3_conv1
    x = vggnet.layers[8](x)         #block3_conv2
    out3 = vggnet.layers[9](x)      #block3_conv3
    x = vggnet.layers[10](out3)        #block3_pool

    x = vggnet.layers[11](x)         #block3_conv1
    x = vggnet.layers[12](x)         #block3_conv2
    out4 = vggnet.layers[13](x)      #block3_conv3

    model = Model(inputs=inputs, outputs=[out1, out2, out3, out4])
    return model


# gram_matric(x) is from keras example neural_style_transfer.py
def gram_matrix(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x[0,:,:,:])
    else:
        features = K.batch_flatten(K.permute_dimensions(x[0,:,:,:], (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

img_nrows = 256
img_ncols = 256

def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def allLoss(y_true, y_pred):
    l_feat = K.mean(K.square(y_pred[5] - y_true[5]), axis=[0,1,2])
    
    l_style = K.sum(K.square(gram_matrix(y_true[1])-gram_matrix(y_pred[1]))) / ((256*256*64)**2)
    l_style += K.sum(K.square(gram_matrix(y_true[2])-gram_matrix(y_pred[2]))) / ((128*128*128)**2)
    l_style += K.sum(K.square(gram_matrix(y_true[3])-gram_matrix(y_pred[3]))) / ((64*64*256)**2)
    l_style += K.sum(K.square(gram_matrix(y_true[4])-gram_matrix(y_pred[4]))) / ((32*32*512)**2)

    return l_feat + l_style + 1e-5*total_variation_loss(y_pred[0])

def featureLoss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[0,1,2])

def styleLoss1(y_true, y_pred):
    return K.sum(K.square(gram_matrix(y_true)-gram_matrix(y_pred))) / ((256*256*64)**2)

def styleLoss2(y_true, y_pred):
    return K.sum(K.square(gram_matrix(y_true)-gram_matrix(y_pred))) / ((128*128*128)**2)

def styleLoss3(y_true, y_pred):
    return K.sum(K.square(gram_matrix(y_true)-gram_matrix(y_pred))) / ((64*64*256)**2)

def styleLoss4(y_true, y_pred):
    return K.sum(K.square(gram_matrix(y_true)-gram_matrix(y_pred))) / ((32*32*512)**2)

def TVLoss(y_true, y_pred):
    return total_variation_loss(y_pred)




"""
width = 256
height = 256

vgg_net = getVggNet(height, width)

whole_net = getWholeNet(256,256)
whole_net.summary()
whole_net.compile(optimizer='adam', loss=allLoss)
result = whole_net.predict([x])

img = image.load_img("test_image/monkey.jpg", target_size=(256,256))
x = image.img_to_array(img)
x = np.expand_dims(x, 0)



#print("result:" + result.shape)
print("result[0]:" + str(result[2].shape))
"""

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