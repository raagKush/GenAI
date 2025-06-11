from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from keras.models import Sequential, Model
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import numpy as np


#Define input image dimensions
#Large images take too much time and resources.
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

##########################################################################
#Given input of noise (latent) vector, the Generator produces an image.
def build_generator():

    noise_shape = (100,) #1D array of size 100 (latent vector / noise)

#Define your generator network 
#Here we are only using Dense layers. But network can be complicated based
#on the application. For example, you can use VGG for super res. GAN.         

    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)    #Generated image

    return Model(noise, img)

#Alpha — α is a hyperparameter which controls the underlying value to which the
#function saturates negatives network inputs.
#Momentum — Speed up the training
##########################################################################

#Given an input image, the Discriminator outputs the likelihood of the image being real.
    #Binary classification - true or false (we're calling it validity)

def build_discriminator():


    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

def train(epochs, batch_size=128, save_interval=50):

    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

#Add channels dimension. As the input to our gen and discr. has a shape 28x28x1.
    X_train = np.expand_dims(X_train, axis=3) 

    half_batch = int(batch_size / 2)

    for epoch in range(epochs):

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]
        noise = np.random.normal(0,1,(half_batch, 100))
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs,np.ones((half_batch,1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs,np.zeros((half_batch,1)))
        
        d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)
        
        noise = np.random.normal(0,1,(batch_size, 100))
        
        valid_y = np.array([1]*batch_size)
        
        g_loss = combined.train_on_batch(noise,valid_y)
        
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        
        if epoch % save_interval == 0:
            save_imgs(epoch, fixed_noise)
            
def save_imgs(epoch,noise):
    r,c = 5,5
    gen_imgs = generator.predict(noise)
    
    gen_imgs = 0.5*gen_imgs + 0.5
    
    fig,axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap = 'gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(r"E:\Workspace\Generative AI\GAN\images\mnist_%d.png" % epoch)
    plt.close()
    
fixed_noise = np.random.normal(0, 1, (25, 100)) 
optimizer = Adam(0.0002,0.5)

discriminator= build_discriminator()
discriminator.compile(loss = 'binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

generator = build_generator()
generator.compile(loss = 'binary_crossentropy', optimizer=optimizer)

z= Input(shape=(100,))
img = generator(z)

discriminator.trainable = False

valid = discriminator(img)

combined = Model(z,valid)
combined.compile(loss = 'binary_crossentropy', optimizer=optimizer)


train(epochs=5000,batch_size=32, save_interval=100)

generator.save(r'E:\Workspace\Generative AI\GAN\generator_model_100k.h5')
