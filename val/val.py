     
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import tensorflow as tf
import cv2
from tensorflow.keras.layers import Activation
import os 

def ssim_loss(y_true, y_pred):
    ssim_loss_value = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))  
    l2_loss_value = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)))
    return 0.5*ssim_loss_value + 0.5*l2_loss_value

def select_image():
    root = Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename() 
    return file_path
def show_data(data, n_imgs=8, title=""):
    plt.figure(figsize=(15, 5))
    n_imgs = min(n_imgs, len(data))  
    for i in range(n_imgs):
        ax = plt.subplot(2, n_imgs, i + 1)
        plt.imshow(image.array_to_img(data[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title(title)
    plt.show()  


model_path = "output/20241106_110105/model_attention.h5"
image_path = select_image()
print(f"Loading model from: {model_path}")
print(f"Type of model_path: {type(model_path)}")


def preprocess_image(image_path, target_size=(448, 320)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def compute_reconstruction_error(image_date):
    results = autoencoder.evaluate(image_date, image_date, batch_size=2)
    print(results)
    formatted_result = f"{results[0]:.3f}"  
    return formatted_result



autoencoder = load_model(model_path, custom_objects={'ssim_loss': ssim_loss, 'Activation': Activation})

input_image = preprocess_image(image_path)

reconstructed = autoencoder.predict(input_image)
# show_data(input_image, title="Original Healthy Images")
# reconstructed_image = autoencoder.predict(input_image)

compute_reconstruction_error(input_image)
# print(reconstructed.shape)  
# print(reconstructed.dtype) 

reconstructed = np.squeeze(reconstructed, axis=0)
reconstructed = np.clip(reconstructed, 0, 1)  
reconstructed = (reconstructed * 255).astype(np.uint8)
cv2.imshow("reconstructed",reconstructed)
cv2.imwrite("dataset/reconstructed_image/reconstructed.jpg",reconstructed)
cv2.waitKey(0)
cv2.destroyAllWindows()
# show_data(reconstructed, title="Reconstructed Images")
