from tensorflow.keras.models import Sequential,Model
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
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acccuracy'])
    model.summary()
    
    return model

def generate_latent_points(latent_dim,n_samples):
    noise = np.random.randn(n_samples,latent_dim)
    return noise
    
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
    gen_output = generator.output #output image
    

    return

def load_normalize_data():
    (X_train,Y_train),(_,_) = cifar10.load_dataset()
    X=X_train.astype('float32')
    X = (X-127.5)/127.5
    
    return X,Y_train

def generate_real_samples(dataset,n_samples):
    idx = np.random.randint(0,dataset.shape[0],size=n_samples)
    X_real,Y_real = dataset[idx]
    
    return X_real,Y_real

def generate_fake_sample(genetator,latent_dim,n_samples):
    X_fake_input,_Y_fake_input = generate_latent_points(latent_dim, n_samples)
    X_fake = 
    
