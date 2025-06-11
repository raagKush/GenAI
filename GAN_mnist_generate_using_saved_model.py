from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import math



# genrate single image
# model = load_model(r'\Test\GAN\mnist\generator_model_100k.h5')

# vector = np.random.randn(100)
# vector = vector.reshape(1,100)

# x= model.predict(vector)

# plt.imshow(x[0,:,:,0],cmap= 'gray_r')
# plt.show()



def plot_images_in_grid(image_list):

    # Calculate the number of images
    num_images = len(image_list)

    # Find the grid dimensions (rows and columns)
    # Start with the square root of the number of images
    rows = int(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)

    # Adjust rows and columns to ensure all images fit
    while rows * cols < num_images:
        rows += 1

    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Plot each image
    for i, img in enumerate(image_list):
        axes[i].imshow(img[0,:,:,0], cmap = 'gray')
        axes[i].axis('off')  # Turn off axis for better visualization

    # Turn off unused subplots
    for j in range(len(image_list), len(axes)):
        axes[j].axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()  


def generate_fake_mnist(vector_size,sample_size):
    generated_image = []
    for i in range(sample_size):
        vector = np.random.randn(vector_size)
        vector = vector.reshape(1,100)
        generated_image.append(model.predict(vector))
    plot_images_in_grid(generated_image)
        
model = load_model(r'\Test\GAN\mnist\generator_model_100k.h5')
generate_fake_mnist(100,9)
