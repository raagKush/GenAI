from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization,Embedding,multiply
from tensorflow.keras.layers import LeakyReLU
from keras.models import Sequential, Model
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import numpy as np
import os,glob


#Define input image dimensions
#Large images take too much time and resources.
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
num_classes= 10
latent_dim = 100

config = {
     "save_image_path" : r"E:\Workspace\Generative AI\Conditional_GAN\mnist\images",
     "checkpoint_path" : r"E:\Workspace\Generative AI\Conditional_GAN\mnist\checkpoints"
 }

save_image_path = config["save_image_path"]
checkpoint_path = config["checkpoint_path"]



##########################################################################
#Given input of noise (latent) vector, the Generator produces an image.
def build_generator():

    noise_shape = (latent_dim,) #1D array of size 100 (latent vector / noise)
    
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
    label = Input(shape=(1,),dtype='int32')

    label_embedding = Flatten()(Embedding(num_classes,latent_dim)(label))
    model_input = multiply([noise,label_embedding])

    img = model(model_input)    #Generated image

    return Model([noise,label], img)

#Alpha — α is a hyperparameter which controls the underlying value to which the
#function saturates negatives network inputs.
#Momentum — Speed up the training
##########################################################################

#Given an input image, the Discriminator outputs the likelihood of the image being real.
    #Binary classification - true or false (we're calling it validity)

def build_discriminator():


    model = Sequential()

    model.add(Dense(512,input_shape=(np.prod(img_shape),)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(1,),dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)
    
    model_input = multiply([flat_img,label_embedding])
    
    
    validity = model(model_input)

    return Model([img,label], validity)

def train(epochs, batch_size=128, image_interval=50, checkpoint_interval = 100):

    # Load the dataset
    (X_train, y_tain), (_, _) = mnist.load_data()

    # Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

#Add channels dimension. As the input to our gen and discr. has a shape 28x28x1.
    X_train = np.expand_dims(X_train, axis=3) 

    half_batch = int(batch_size / 2)

    for epoch in range(start_epoch,epochs):

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs,labels = X_train[idx],y_tain[idx]
        
        noise = np.random.normal(0,1,(half_batch, latent_dim))
        gen_labels = np.random.randint(0,num_classes,half_batch)
        gen_imgs = generator.predict([noise,gen_labels])
        
        d_loss_real = discriminator.train_on_batch([imgs,labels],np.ones((half_batch,1)))
        d_loss_fake = discriminator.train_on_batch([gen_imgs,gen_labels],np.zeros((half_batch,1)))
        
        d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)
        
        noise = np.random.normal(0,1,(batch_size, latent_dim))
        sampled_labels= np.random.randint(0,num_classes,batch_size)
        
        valid_y = np.array([1]*batch_size)
        
        g_loss = combined.train_on_batch([noise,sampled_labels],valid_y)
        
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        
        if epoch % image_interval == 0:
            save_imgs(epoch)
        
        if epoch % checkpoint_interval == 0:
            gen_path = os.path.join(checkpoint_path,f"generator_epoch_{epoch}.h5")
            disc_path = os.path.join(checkpoint_path,f"discriminator_epoch_{epoch}.h5")
            generator.save_weights(gen_path)
            discriminator.save_weights(disc_path)
            print(f"Checkpoint save for epoch: {epoch}")
            
def save_imgs(epoch):
    r,c = 2,5
    
    noise = np.random.normal(0,1,size=(r*c,latent_dim))
    sampled_labels = np.array([i for i in range(10)])
    
    
    gen_imgs = generator.predict([noise,sampled_labels])
    gen_imgs = 0.5*gen_imgs + 0.5
    
    fig,axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap = 'gray')
            axs[i,j].set_title(f"Digit: {sampled_labels[cnt]}")
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(save_image_path, "mnist_%d.png") % epoch)
    plt.close()
    
def find_latest_scheckpoint(model_name):
    pattern = os.path.join(checkpoint_path, f"{model_name}_epoch_*.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        return None,0
    latest = files[-1]                          # e.g. ".../generator_epoch_120.h5"
    basename = os.path.basename(latest)         # "generator_epoch_120.h5"
    name_only, _ = os.path.splitext(basename)  # "generator_epoch_120"
    parts = name_only.split("_")                # ["generator", "epoch", "120"]
    epoch = int(parts[-1])
    return latest,epoch


    
os.makedirs(checkpoint_path,exist_ok=True)
gen_chkpt,gen_start = find_latest_scheckpoint("generator")
disc_chkpt,disc_start = find_latest_scheckpoint("discriminator")
start_epoch = max(gen_start,disc_start)

optimizer = Adam(0.0002,0.5)

discriminator= build_discriminator()
discriminator.compile(loss = 'binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

generator = build_generator()
generator.compile(loss = 'binary_crossentropy', optimizer=optimizer)

if gen_chkpt:
    print("Loading generator weights")
    generator.load_weights(gen_chkpt)
    
if disc_chkpt:
    print("Loading discriminator weights")
    discriminator.load_weights(disc_chkpt)

z= Input(shape=(latent_dim,))
label = Input(shape=(1,))
img = generator([z,label])

discriminator.trainable = False

valid = discriminator([img,label])

combined = Model([z,label],valid)
combined.compile(loss = 'binary_crossentropy', optimizer=optimizer)


train(epochs=5000,batch_size=32, image_interval=100, checkpoint_interval= 400)

generator.save(r'E:\Workspace\Generative AI\Conditional_GAN\mnist\generator_conditional_5000.h5')
