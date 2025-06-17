from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape, Embedding, Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers.legacy import Adam

img_size = (32,32,3)
def define_discriminator(input_size=img_size):
    
    model = Sequential()
    model.add(Conv2D(128, (3,3), padding='same',input_shape=img_size))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    
    
    opt = Adam(learning_rate=0.002,beta_1=0.5)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acccuracy'])
    model.summary()
    
    return model

    
def define_generator(latent_dim,label):
    
    embedding = Embedding(input_dim=10,output_dim=50)(label)
    embedding = embedding.Flatten()
    
    model = Sequential()
    
    return
    
def degine_gan():
    return

def load_normalize_data():
    return

    
