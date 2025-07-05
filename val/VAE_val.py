import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

def preprocess_image(file_path, target_size=(448, 320)):
    img = Image.open(file_path).resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0
    return img_array

def compute_reconstruction_error(model, image_data):
    results = model.evaluate(image_data, image_data, batch_size=2)
    print(f"重建损失: {results}")

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def process_images_in_folder(folder_path):
    all_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg')):
            file_path = os.path.join(folder_path, filename)
            img_array = preprocess_image(file_path)
            all_images.append(img_array)
    
    all_images = np.array(all_images)
    compute_reconstruction_error(vae, all_images)

# 加载模型
vae = load_model('output/vae_model.h5', custom_objects={'sampling': sampling})

# 处理文件夹中的图片
folder_path ="dataset/val/normal"  
process_images_in_folder(folder_path)
