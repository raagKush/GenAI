from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Reshape, Concatenate, LeakyReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import  RandomNormal
from tensorflow.keras.layers import Activation

import numpy as np
from  matplotlib import pyplot as plt
import os
from tensorflow.keras.utils import load_img,img_to_array

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

def define_GAN(g_model,d_model,image_shape):
    
    '''Instead of making whole d_model nontrainable, only Batchnormalization 
    is set to train.
    BatchNormalization behaves differently during training and inference:
        It keeps running averages of mean and variance during training.
        If set to non-trainable, those stats will not update, which may lead to training instability.
    GANs are sensitive to normalization behavior, and freezing BatchNorm can cause degradation in the training dynamics.
    '''
    
    for layer in d_model.layers:
        if not isinstance(layer,BatchNormalization):
            layer.trainable = False
            
    in_src = Input(shape=image_shape)
    
    gen_out = g_model(in_src)
    dis_out = d_model([in_src,gen_out])
    
    model = Model(in_src,[dis_out,gen_out])
    
    opt = Adam(learning_rate=0.0002,beta_1=0.5)
    
    model.compile(optimizer=opt,loss=['binary_crossentropy','mae'],loss_weights=[1,100])
    return model    
    
def generate_real_samples(dataset,n_samples,patch_shape):
    
    trainA,trainB = dataset #TrainA - satellite image, trainB - corresponding maps
    idx = np.random.randint(0,trainA.shape[0],n_samples)
    X1,X2 = trainA[idx], trainB[idx]
    y = np.ones((n_samples,patch_shape,patch_shape,1))
    return [X1,X2],y

def generate_fake_samples(g_model,samples,patch_shape):
    X = g_model(samples)
    
    y = np.zeros((len(X),patch_shape,patch_shape,1))
    
    return X,y

def summarize_performance_save_model(step,g_model,dataset, n_samples=3):
    [X_realA,X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    X_fakeB = generate_fake_samples(g_model, X_realA, 1)
    
    #scale back pixels
    
    X_realA = (X_realA+1)/2.0
    X_realB = (X_realB+1)/2.0
    X_fakeB = (X_fakeB+1)/2.0
    
    for i in range (n_samples):
        plt.subplot(3, n_samples,1+i)
        plt.axis('off')
        plt.imshow(X_realA[i])
        
    for i in range (n_samples):
        plt.subplot(3, n_samples,1+n_samples+i)
        plt.axis('off')
        plt.imshow(X_realB[i])
        
    for i in range (n_samples):
        plt.subplot(3, n_samples,1+n_samples*2+i)
        plt.axis('off')
        plt.imshow(X_fakeB[i])

    filename_plt = r'\WORKSPACE\Test\GAN\pix2pix\result\plot_%06d.png' % (step+1)
    plt.savefig(filename_plt)
    plt.close()
    
    filename_model = r'\WORKSPACE\Test\GAN\pix2pix\result\model_%06d.h5' % (step+1)
    g_model.save(filename_model)
    
    print('>Saved model at {}'.format(filename_model))

def train(d_model,g_model,gan_model,dataset, num_epochs=20, batchSize = 1):
    
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    batch_per_epoch = int(len(trainA)/num_epochs)
    n_steps = batch_per_epoch*num_epochs
    
    for i in range(n_steps):
            
        [X_realA, X_realB], y_real = generate_real_samples(dataset, batchSize, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        
        d_loss2 = d_model.train_on_batch([X_realA,X_fakeB], y_fake)
        
        g_loss,_,_ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        
        print(">Epoch{}, d_loss1 = {}, d_loss2={}, g_loss = {}".format(i+1, d_loss1, d_loss2, g_loss))
        
        if (i+1)%(batch_per_epoch*10) == 0:
            summarize_performance_save_model(i, g_model, dataset)
            
        
from PIL import Image

def load_images(path, size=(512,256)):
    src_list, target_list = [], []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if not os.path.isfile(file_path) or not filename.lower().endswith(valid_exts):
            continue

        image = Image.open(file_path).convert("RGB")
        image = image.resize(size)  # size = (width, height) for PIL!
        print("PIL image size (width, height):", image.size)  # Debug

        combined_img = img_to_array(image)
        print("NumPy array shape (height, width, channels):", combined_img.shape)  # Debug

        # Now split width-wise at 256 pixels:
        sat_img = combined_img[:, :256, :]
        map_img = combined_img[:, 256:, :]

        src_list.append(sat_img)
        target_list.append(map_img)

    return [np.asarray(src_list), np.asarray(target_list)]
        
        
path = r'\WORKSPACE\Test\GAN\pix2pix\maps\train'
        
[src_images, target_images] = load_images(path)
print("Dataset loaded")
n_samples = 4

for i in range(n_samples):
    plt.subplot(2,n_samples,1+i)
    plt.axis('off')
    plt.imshow(src_images[i].astype('uint8'))
        
for i in range(n_samples):
    plt.subplot(2,n_samples,1+n_samples+i)
    plt.axis('off')
    plt.imshow(target_images[i].astype('uint8'))
    
