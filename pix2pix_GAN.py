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

def define_generator(image_shape=(256,256,3)):
    
    init = RandomNormal(stddev=0.02)
    
    in_image = Input(shape=image_shape)
    
    #encoder
    
    e1 = define_encoder_block(in_image,64,batchNorm=False)
    e2 = define_encoder_block(e1,128)
    e3 = define_encoder_block(e2,256)
    e4 = define_encoder_block(e3,512)
    e5 = define_encoder_block(e4,512)
    e6 = define_encoder_block(e5,512)
    e7 = define_encoder_block(e6,512)
    
    #bottleneck, without batchnorm and WITH relu instead of leakyrelu
    
    b = Conv2D(512,(4,4),padding='same',kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    
    #decoder
    
    d1 = define_decoder_block(b,e7,512)
    d2 = define_decoder_block(d1,e6,512)
    d3 = define_decoder_block(d2,e5,512)
    d4 = define_decoder_block(d3,e4,512,dropout=False)
    d5 = define_decoder_block(d4,e3,256,dropout=False)
    d6 = define_decoder_block(d5,e2,128,dropout=False)
    d7 = define_decoder_block(d6,e1,64,dropout=False)
    
    #output
    
    g = Conv2DTranspose(image_shape[2],(4,4),padding='same',kernel_initializer=init)(d7)
    
    out_image= Activation('tanh')(g)
    
    model = Model(in_image,out_image)
    
    return model
