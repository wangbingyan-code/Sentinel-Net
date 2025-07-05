import os
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model_path = "output/20241106_110105/model_attention.h5"
folder_path = "dataset/train/anomaly"  
def ssim_loss(y_true, y_pred):
    ssim_loss_value = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    # l2_loss_value = tf.reduce_mean(tf.square(y_true - y_pred))  
    l2_loss_value = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)))
    return 0.5*ssim_loss_value + 0.5*l2_loss_value

def preprocess_image(image_path, target_size=(448, 320)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # 增加批处理维度
    return img_array

# 计算并显示重建误差
def compute_reconstruction_error(image_data):
    reconstruct_loss = autoencoder.evaluate(image_data, image_data, batch_size=2, verbose=0)
    return reconstruct_loss[0]
    

# 加载模型
autoencoder = load_model(model_path, custom_objects={'ssim_loss': ssim_loss})

# 为文件夹下的每一张图片创建一个路径
image_paths = [os.path.join(folder_path, image_file) for image_file in os.listdir(folder_path) if image_file.endswith((".png", ".jpg"))]
total_files = len(image_paths)
anomaly_count = 0  # 异常文件计数

for image_path in image_paths:
    print(f"处理图片 {image_path}")
    input_image = preprocess_image(image_path)
    reconstruct_loss = compute_reconstruction_error(input_image)
    reconstruct_loss = float(reconstruct_loss)
    if reconstruct_loss > 0.15:
        anomaly_count += 1
        print(f"异常图片: {image_path} | 重建误差: {reconstruct_loss}")
anomaly_percentage = (anomaly_count / total_files) * 100
print(f"总文件数: {total_files}, 异常文件数: {anomaly_count}, 异常占比: {anomaly_percentage:.2f}%")