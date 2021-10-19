
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, UpSampling2D, Reshape, LeakyReLU, ReLU
from tensorflow.keras import regularizers
import tensorflow as tf

def first_conv(X, layer_size , w, h, c, sparsity_str):
    #conv_layer = Conv2D(layer_size, kernel_size=(5,5), input_shape=(w, h, c), padding ='same', activation = 'relu' , activity_regularizer=regularizers.l1(sparsity_str))(X)
    #return(conv_layer)
    conv_layer = Conv2D(layer_size, kernel_size=(5,5), input_shape=(w, h, c), padding ='same',  activity_regularizer=regularizers.l1(sparsity_str))(X)
    #act_layer = LeakyReLU()(conv_layer)
    act_layer = ReLU()(conv_layer)
    return(act_layer)

def conv(X, layer_size, sparsity_str):
    conv_layer = Conv2D(layer_size, kernel_size=(5,5), padding='same', activity_regularizer=regularizers.l1(sparsity_str))(X)
    #act_layer = LeakyReLU()(conv_layer)
    act_layer = ReLU()(conv_layer)
    #return(conv_layer)
    return(act_layer)

def maxpool(X):
    max_pool_layer = MaxPooling2D()(X)
    return(max_pool_layer)

def flatten(X):
    flatten_layer = Flatten()(X)
    return(flatten_layer)

def dense(X, latent_size, activation_func):
    dense_layer = Dense(latent_size, activation=activation_func)(X)
    return(dense_layer)

def reshape(X, dim1, dim2, dim3):
    reshape_layer = Reshape((dim1,dim2,dim3))(X)
    return(reshape_layer)

def unpool(X):
    unpool_layer = UpSampling2D()(X)
    return(unpool_layer)

def deconv(X, layer_size, sparsity_str):
    deconv_layer = Conv2DTranspose(layer_size, kernel_size=(5,5), padding='same', activation='relu', activity_regularizer=regularizers.l1(sparsity_str))(X)
    return(deconv_layer)

def last_deconv(X, layer_size, sparsity_str):
    deconv_layer = Conv2DTranspose(layer_size, kernel_size=(5,5), padding='same', activation='sigmoid', activity_regularizer=regularizers.l1(sparsity_str))(X)
    return(deconv_layer)

def batchnormalization(X):
    batchnorm_layer = BatchNormalization()(X)
    return(batchnorm_layer)

def dropout(X, drop_perc):
    dropout_layer = Dropout(drop_perc)(X)
    return(dropout_layer)
    