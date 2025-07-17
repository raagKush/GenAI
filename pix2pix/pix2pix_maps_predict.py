from tensorflow.keras import models
import os
from PIL import Image
from tensorflow.keras.utils import img_to_array
import numpy as np
from matplotlib import pyplot as plt

model = models.load_model(r"\WORKSPACE\Test\GAN\pix2pix\result\model_032880.h5")

model.summary()

ValFolderPath = r"\WORKSPACE\Test\GAN\pix2pix\maps\val"

def load_images(path, size=(512,256)):
    src_list, target_list = [], []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if not os.path.isfile(file_path) or not filename.lower().endswith(valid_exts):
            continue

        image = Image.open(file_path).convert("RGB")
        image = image.resize(size)  # size = (width, height) for PIL!


        combined_img = img_to_array(image)

        # Now split width-wise at 256 pixels:
        sat_img = combined_img[:, :256, :]
        map_img = combined_img[:, 256:, :]

        src_list.append(sat_img)
        target_list.append(map_img)

    return [np.asarray(src_list), np.asarray(target_list)]

img_list, map_list = load_images(ValFolderPath)

def preprocess_data(data):
    X1,X2 = data[0], data[1]
    
    X1 = (X1-127.5)/127.5
    X2 = (X2-127.5)/127.5
    
    return [X1,X2]

data = [img_list,map_list]
img_list_nor, map_list_norm = preprocess_data(data)


def predict_maps(img):
    input_tensor = np.expand_dims(img, axis=0)
    predicted = model.predict(input_tensor)
    output = (predicted[0]+1)/2

    return output

# samples = 3
# idx = np.random.randint(0,len(img_list),samples)

# for i in range(len(idx)):
#     plt.subplot(samples,3, i+1)
#     plt.imshow(img_list[idx[i]].astype('uint8'))
#     plt.axis('off')
    
#     plt.subplot(samples,3, 1+samples + i)
#     plt.imshow(map_list[idx[i]].astype('uint8'))
#     plt.axis('off')
    
#     input_to_model = img_list_nor[idx[i]]
#     predicted_img = predict_maps(input_to_model)
#     plt.subplot(samples,3, 1+samples*2+i)
#     plt.imshow(predicted_img)
#     plt.axis('off')
    
# plt.tight_layout()
# plt.show()    

samples = 3
idx = np.random.randint(0,len(img_list),samples)
plt.figure(figsize=(12, 4 * samples))  # Width × Height

for i in range(samples):
    sample_idx = idx[i]

    # Column 1: Input Image
    plt.subplot(samples, 3, i * 3 + 1)
    plt.imshow(img_list[sample_idx].astype('uint8'))
    plt.axis('off')
    if i == 0:
        plt.title("Input")

    # Column 2: Ground Truth / Map
    plt.subplot(samples, 3, i * 3 + 2)
    plt.imshow(map_list[sample_idx].astype('uint8'))
    plt.axis('off')
    if i == 0:
        plt.title("Target")

    # Column 3: Predicted Image
    input_to_model = img_list_nor[idx[i]]
    predicted_img = predict_maps(input_to_model)

    plt.subplot(samples, 3, i * 3 + 3)
    plt.imshow(predicted_img)
    plt.axis('off')
    if i == 0:
        plt.title("Predicted")

plt.tight_layout()
plt.show()





