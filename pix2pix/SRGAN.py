from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,BatchNormalization, PReLU, add, UpSampling2D

def fetch_vgg(hr_shape):
    vgg= VGG19(weights="imagenet",include_top = False, input_shape = hr_shape)
    
    return Model(inputs = vgg.inputs, outputs = vgg.layers[10].output)

def define_resnet(ip):
    
    res_model = Conv2D(64,(3,3),padding = 'same')(ip)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes = [1,2])(res_model)
    
    res_model = Conv2D(64,(3,3),padding = 'same')(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    
    return add([ip,res_model])
