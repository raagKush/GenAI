from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Reshape, Concatenate, LeakyReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import  RandomNormal
from tensorflow.keras.layers import Activation

import numpy as np

def define_discriminator(image_shape): #C64-C128-C256-C512
    
    init = RandomNormal(stddev=0.02)
    
    input_src_image = Input(shape=image_shape)
    
    input_target_image = Input(shape=image_shape)
    
    merged = Concatenate()([input_src_image,input_target_image])
    
    d = Conv2D(64,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(128,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(256,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)    
    
    d = Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(512,(4,4),padding='same',kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(1,(4,4),padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    
    model = Model([input_src_image,input_target_image],patch_out)
    
    opt = Adam(learning_rate=0.0002,beta_1=0.5)
    
    model.compile(optimizer=opt, loss='binary_crossentropy', loss_weights=[0.5])
    model.summary()
    return model

#disc = define_discriminator((256,256,3))

def define_encoder_block(layer_in,num_filters, batchNorm = 'True'):
    
    init = RandomNormal(stddev=0.2)
    
    enc = Conv2D(num_filters,(4,4),strides=(2,2),padding ='same',kernel_initializer=init)(layer_in)
    if batchNorm:
        enc = BatchNormalization()(enc,training=True)
    enc = LeakyReLU(alpha=0.2)(enc)
    
    return enc
    
def define_decoder_block(layer_in,skip_in,num_filters,dropout=True):
    
    init = RandomNormal(stddev=0.2)
    
    dec = Conv2DTranspose(num_filters,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(layer_in)
    dec = BatchNormalization()(dec,training=True)
    if dropout:
        dec = Dropout(0.5)(dec, training=True)
    dec = Concatenate()([dec,skip_in])
    dec = Activation('relu')(dec)
    
    return dec

