import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, Dense, Lambda, Dropout, Reshape
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.font_manager as fm
import pandas as pd
import time
from tensorflow.keras.losses import MeanSquaredError


def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def sampling(args):
    """通过重参数化技巧采样潜在变量 z."""
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
def set_chinese_font():
    font_path = "C:/Windows/Fonts/simhei.ttf"  
    if os.path.exists(font_path):
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
    else:
        print("字体文件未找到，请调整路径")

set_chinese_font()

def create_dataset(directory, batch_size=32, target_size=(448, 320)):
    filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]

    def load_and_preprocess_image(filename):
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = img / 255.0  # 归一化到 [0, 1]
        return img, img

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

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

def build_vae(input_shape=(448, 320, 3), latent_dim=128, activation='relu'):
    # 编码器部分
    input_layer = Input(shape=input_shape, name="INPUT")
    x = layers.GaussianNoise(stddev=0.2)(input_layer)

    if activation == 'leaky_relu':
        activation_fn = layers.LeakyReLU(alpha=0.2)
    elif activation == 'elu':
        activation_fn = layers.ELU(alpha=1.0)
    elif activation == 'prelu':
        activation_fn = layers.PReLU()
    elif activation == 'swish':
        activation_fn = tf.keras.activations.swish
    elif activation == 'selu':
        activation_fn = tf.keras.activations.selu
    else:
        activation_fn = layers.Activation(activation)

    x = Conv2D(128, (5, 5), padding='same')(x)
    x = activation_fn(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = activation_fn(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (5, 5), padding='same')(x)
    x = activation_fn(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), padding='same')(x)
    x = activation_fn(x)
    x = Dropout(0.3)(x)
    encoded = MaxPooling2D((2, 2))(x)

    encoder_shape = tf.keras.backend.int_shape(encoded)[1:]
    x = Flatten()(encoded)
    x = Dense(512)(x)
    x = activation_fn(x)

    # 均值和对数方差
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

    # 解码器部分
    latent_inputs = Input(shape=(latent_dim,), name="latent_inputs")
    x = Dense(tf.reduce_prod(encoder_shape))(latent_inputs)
    x = activation_fn(x)
    x = Reshape(encoder_shape)(x)

    x = Conv2DTranspose(16, (3, 3), padding='same')(x)
    x = activation_fn(x)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(32, (5, 5), padding='same')(x)
    x = activation_fn(x)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(64, (3, 3), padding='same')(x)
    x = activation_fn(x)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(128, (5, 5), padding='same')(x)
    x = activation_fn(x)
    x = Dropout(0.3)(x)
    x = UpSampling2D((2, 2))(x)

    output_layer = Conv2D(3, (3, 3), padding='same', name="OUTPUT")(x)

    # 定义模型
    encoder = Model(input_layer, [z_mean, z_log_var, z], name="encoder")
    decoder = Model(latent_inputs, output_layer, name="decoder")
    outputs = decoder(encoder(input_layer)[2])
    vae = Model(input_layer, outputs, name="vae")

    # 自定义损失函数
    reconstruction_loss = MeanSquaredError()(input_layer, outputs)
    reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2]

    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)

    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder, decoder

# 数据集路径
traindir = "../dataset/train/normal"
normal_dir = "../dataset/val/normal"
anomaly_dir = "../dataset/val/anomaly"

train_dataset = create_dataset(traindir, batch_size=4)
normal_dataset = create_dataset(normal_dir, batch_size=1)
anomaly_dataset = create_dataset(anomaly_dir, batch_size=1)

train_size = len([name for name in os.listdir(traindir) if name.endswith(('png', 'jpg'))])
normal_size = len([name for name in os.listdir(normal_dir) if name.endswith(('png', 'jpg'))])
anomaly_size = len([name for name in os.listdir(anomaly_dir) if name.endswith(('png', 'jpg'))])

print(f"训练数据集大小: {train_size}")
print(f"正常数据集大小: {normal_size}")
print(f"异常数据集大小: {anomaly_size}")

# 构建 VAE 模型
vae, encoder, decoder = build_vae(input_shape=(448, 320, 3), latent_dim=128, activation='relu')

choice = input("您想训练一个新模型还是加载一个现有模型？(train/load): ").strip().lower()

if choice == 'train':
    vae.summary()
    history = vae.fit(train_dataset, epochs=100, validation_data=normal_dataset)
    vae.save('../output/vae_model.h5') 
    loss_data = history.history['loss']
    df = pd.DataFrame({'epoch': range(1, len(loss_data) + 1), 'loss': loss_data})
    df.to_csv('../output/loss_data_vae.csv', index=False)
    
    plt.plot(history.history['loss'], label='Loss') 
    plt.title('Model Loss', fontsize=16)  
    plt.ylabel('Loss', fontsize=14)  
    plt.xlabel('Epochs', fontsize=14)  
    plt.legend(fontsize=12)  
    plt.grid() 
    plt.show()

else:
    
    vae = load_model(
    '../output/vae_model.h5')
vae.summary()

normal_data_batch = next(iter(normal_dataset))[0]
start_time = time.time()
reconstructed = vae.predict(normal_data_batch)
end_time = time.time()  
inference_time = end_time - start_time
print(f"normal_dataset 数据集推理耗时: {inference_time:.4f} 秒，平均每张图片耗时: {inference_time / normal_size:.4f} 秒")
# show_data(normal_data_batch, title="原始正常图片")
show_data(reconstructed, title="重建正常图片")

results = vae.evaluate(normal_dataset, steps=normal_size)
print("正常数据的损失和准确率", results)

anomaly_data_batch = next(iter(anomaly_dataset))[0]
start_time = time.time()
reconstructed = vae.predict(anomaly_data_batch)
end_time = time.time()  
inference_time = end_time - start_time
print(f"anomaly_dataset 数据集推理耗时: {inference_time:.4f} 秒，平均每张图片耗时: {inference_time / anomaly_size:.4f} 秒")

show_data(anomaly_data_batch, title="原始异常图片")
show_data(reconstructed, title="重建异常图片")

results = vae.evaluate(anomaly_dataset, steps=anomaly_size)
print("异常数据的损失和准确率", results)