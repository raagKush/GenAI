from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.datasets import cifar10
import numpy as np
from matplotlib import pyplot as plt
import keras

def define_discriminator(in_shape = (32,32,3)):
    model = Sequential()
    model.add(Conv2D(128,(3,3), padding='same', strides=(2,2), input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128,(3,3), padding='same', strides=(2,2)))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Flatten())   
    model.add(Dropout(0.4))
    model.add(Dense(1,activation='sigmoid'))
    
    opt = Adam(lr = 0.002, beta_1=0.5)
    model.compile(optimizer=opt,metrics=['accuracy'],loss='binary_crossentropy')
    
    return model

def define_generator(latent_dim=100):
    model =Sequential()
    
    n_nodes = 8*8*128
    
    model.add(Dense(n_nodes,input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8,8,128)))
    
    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(3,(8,8),activation='tanh',padding='same',))
    
    return model

def define_gan(generator,discriminator):
    discriminator.trainable = False
    
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    
    opt = Adam(lr=0.002,beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    
    return model

def load_normalize_real_data(): 
    (X_train,_),(_,_) = cifar10.load_data()
    X= X_train.astype('float32')
    
    X = (X-127.5)/127.5 # -1 to 1 as we are using tanh
    
    return X
    
def generate_real_samples(dataset,n_samples):
    
    idx = np.random.randint(0,dataset.shape[0],size=n_samples)

    X_real = dataset[idx]
    y_real = np.ones((n_samples,1))
       
    return X_real,y_real


def generate_latent_points(latent_dim,n_samples):
    noise = np.random.randn(n_samples,latent_dim)
    return noise
    
def generate_fake_samples(generator, latent_dim, n_samples):
    x_fake_input = generate_latent_points(latent_dim, n_samples)
    X_fake = generator.predict(x_fake_input)
    y_fake = np.zeros((n_samples,1))   
    
    return X_fake,y_fake

    
def train_gan(generator,discriminator,gan_model,dataset,latent_dim=100,epochs=500,batch_size=128):
    batch_per_epoch = int(dataset.shape[0]/batch_size)
    half_batch = int(batch_size/2)
    
    for i in range(epochs):
        for j in range(batch_per_epoch):
            
            # Step 1: Train Discriminator on Real Data
            X_real, y_real = generate_real_samples(dataset,half_batch)
            d_loss_real,_ = discriminator.train_on_batch(X_real,y_real)
            
            # Step 2: Train Discriminator on Fake Data
            X_fake, y_fake = generate_fake_samples(generator,latent_dim,half_batch)
            d_loss_fake,_ = discriminator.train_on_batch(X_fake,y_fake)
            
            # Step 3: Train generator
            X_gan = generate_latent_points(latent_dim,batch_size)
            y_gan = np.ones((batch_size,1))
            
            g_loss = gan_model.train_on_batch(X_gan,y_gan)
            
            print("Epoch: {}, Batch: {}/{}, g_loss = {}, d_loss = {}".format(i+1,j+1,batch_per_epoch,g_loss,(d_loss_real+d_loss_fake)/2))
            
    generator.save(r'C:\WORKSPACE\Test\GAN\cifar10\generator_model.h5')
       
latent_dim =100    
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator,discriminator)
dataset = load_normalize_real_data()

train_gan(generator,discriminator,gan_model,dataset,latent_dim,epochs=50,batch_size=128)
