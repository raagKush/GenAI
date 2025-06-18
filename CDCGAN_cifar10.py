from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape, Embedding, Input, Concatenate, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.datasets import cifar10
import numpy as np

input_size = (32,32,3)
def define_discriminator(input_size=input_size,num_classes=10):
    
    in_label = Input(shape=(1,),dtype='int32')
    label_embedding = Embedding(num_classes,50)(in_label)    
    n_nodes = input_size[0]*input_size[1] #32x32
    label_embedding = Dense(n_nodes)(label_embedding)
    label_embedding = Reshape((input_size[0],input_size[1],1))(label_embedding) #32x32x1 (Assusme Embedding as additional channel)
    
    in_image = Input(shape=input_size)
    
    merge = Concatenate()([in_image,label_embedding]) #concatenate 32x32x3 with 32x32x1 to make it 32x32x4
    
    conv1 = Conv2D(128, (3,3), padding='same')(merge)
    leakyRelu1= LeakyReLU(alpha=0.2)(conv1)
    
    conv2 = Conv2D(128, (3,3), padding='same')(leakyRelu1)
    leakyRelu2= LeakyReLU(alpha=0.2)(conv2)
    
    flat = Flatten()(leakyRelu2)
    drop = Dropout(0.4)(flat)
    
    outLayer = Dense(1,activation='sigmoid')(drop)
    
    model = Model([in_image,in_label],outLayer) #Model input[in1,in2] and output[out]
    
    
    opt = Adam(learning_rate=0.002,beta_1=0.5)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    
    return model
    
def define_generator(latent_dim=100, num_classes = 10):
    
    in_label = Input(shape=(1,))
    label_embedding = Embedding(num_classes,50)(in_label)
    n_nodes = 8*8
    label_embedding = Dense(n_nodes)(label_embedding)
    label_embedding = Reshape((8,8,1))(label_embedding)
    in_latent_vector = Input(shape=(latent_dim,))
    
    
    n_nodes = 8*8*128 #start with 8x8 and upscale to 32x32 image
    Dense1 = Dense(n_nodes)(in_latent_vector)
    LeakyRelu1= LeakyReLU(alpha=0.2)(Dense1)
    Reshaped = Reshape((8,8,128))(LeakyRelu1)
    
    merge = Concatenate()([Reshaped,label_embedding])
    
    
    conv1 = Conv2DTranspose(128,(4,4),strides=(2,2),padding='same')(merge)
    LeakyRelu2 = LeakyReLU(alpha=0.2)(conv1)
    
    conv2 = Conv2DTranspose(128,(4,4),strides=(2,2),padding='same')(LeakyRelu2)
    LeakyRelu3 = LeakyReLU(alpha=0.2)(conv2)
    
    outLayer = Conv2D(3,(8,8),padding='same',activation='tanh')(LeakyRelu3)
    
    model = Model([in_latent_vector,in_label],outLayer)
        
    return model


    
def define_gan(discriminator,generator):
    discriminator.trainable = False
    
    gen_noise,gen_labels = generator.input #input of generator model i.e. in_latent_vector,in_label
    gen_output = generator.output #output image produced by generator
    
    gan_output = discriminator([gen_output,gen_labels]) #image with label sent to discriminator to tell if it's real or fake
    
    model = Model([gen_noise,gen_labels],gan_output)
    
    opt = Adam(learning_rate=0.002, beta_1=0.5)
    
    model.compile(loss='binary_crossentropy', optimizer=opt)
    
    return model

def load_normalize_data():
    (X_train,Y_train),(_,_) = cifar10.load_data()
    X=X_train.astype('float32')
    X = (X-127.5)/127.5
    
    return [X,Y_train]

def generate_real_samples(dataset,n_samples):
    idx = np.random.randint(0,dataset[0].shape[0],size=n_samples)
    X_real = dataset[0][idx]
    Y_real = dataset[1][idx]
    
    return X_real,Y_real

def generate_latent_points(latent_dim,n_samples,n_classes=10):
    noise = np.random.randn(n_samples,latent_dim)
    labels = np.random.randint(0,n_classes,n_samples,)
    return noise,labels

def generate_fake_sample(generator, latent_dim, n_samples):
    input_noise,label_fake = generate_latent_points(latent_dim, n_samples)
    
    X_fake = generator.predict([input_noise,label_fake])
    
    return X_fake,label_fake
    

def train(discriminator,generator, gan_model, dataset, latent_dim, batch_size, epochs):
    half_batch = int(batch_size/2)
    batch_per_epoch = int(dataset[0].shape[0]/batch_size) #An epoch is one full pass through the entire training dataset.
                                            #If you have 10,000 images and use batch_size=64, then:
                                            #Number of batches per epoch = 10,000 / 64 ≈ 157 updates.
                                            
    for i in range(epochs):
        for j in range(batch_per_epoch):
            
            X_real,y_real = generate_real_samples(dataset, half_batch)
            d_loss_real = discriminator.train_on_batch([X_real,y_real],np.ones((half_batch,1)))
            
            X_fake,y_fake = generate_fake_sample(generator, latent_dim, half_batch)
            d_loss_fake = discriminator.train_on_batch([X_fake,y_fake],np.zeros((half_batch,1)))
            
            gan_input, gan_label = generate_latent_points(latent_dim, batch_size)
            gan_loss = gan_model.train_on_batch([gan_input, gan_label],np.zeros((batch_size,1)))
            
            print("Epoch:{}/{} Batch{}/{}, d_loss_real:{}, d_loss_fake:{}, gan_loss:{}".format(i,epochs,j,batch_per_epoch,d_loss_real,d_loss_fake,gan_loss))
            
    gan_model.save(r'E:\Workspace\Generative AI\Conditional_GAN\cifar10\generator_conditional.h5')
    
latent_dim = 100
num_classes = 10
dataset = load_normalize_data()
discriminator = define_discriminator(input_size=(32,32,3),num_classes=num_classes)
generator = define_generator(latent_dim,num_classes)
gan_model = define_gan(discriminator, generator)

train(discriminator, generator, gan_model, dataset, latent_dim, 128, 5)
