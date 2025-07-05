import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, Dense, Lambda, Dropout,Reshape
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.font_manager as fm

def set_chinese_font():
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 根据系统字体路径进行调整
    if os.path.exists(font_path):
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
    else:
        print("字体文件未找到，请调整路径")

set_chinese_font()

def data_generator(directory, batch_size=32, target_size=(448, 320)):
    filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]
    while True:
        np.random.shuffle(filenames)
        for i in range(0, len(filenames), batch_size):
            batch_filenames = filenames[i:i+batch_size]
            batch_images = []
            for file in batch_filenames:
                img = image.load_img(file, target_size=target_size)
                img_array = image.img_to_array(img)
                batch_images.append(img_array)
            batch_images = np.array(batch_images).astype('float32') / 255.0
            yield batch_images, batch_images

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

def sampling(args):
    """重参数化技巧"""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_variational_autoencoder(input_shape=(448, 320, 3), latent_dim=512, activation='relu'):
    input_layer = Input(shape=input_shape, name="INPUT")
    
    encoder = Conv2D(128, (3, 3), padding='same')(input_layer)
    encoder = Dropout(0.3)(encoder)
    encoder = MaxPooling2D((2, 2))(encoder)

    encoder = Conv2D(64, (3, 3), padding='same')(encoder)
    encoder = Dropout(0.3)(encoder)
    encoder = MaxPooling2D((2, 2))(encoder)

    encoder = Conv2D(32, (3, 3), padding='same')(encoder)
    encoder = Dropout(0.3)(encoder)
    encoder = MaxPooling2D((2, 2))(encoder)

    # 获取编码后的形状
    encoder_shape = tf.keras.backend.int_shape(encoder)[1:]

    # 平坦化
    encoder_flat = Flatten()(encoder)
    
    # 均值和对数方差
    z_mean = Dense(latent_dim)(encoder_flat)
    z_log_var = Dense(latent_dim)(encoder_flat)

    # 采样
    z = Lambda(sampling)([z_mean, z_log_var])

    # 解码器
    decoder_input = Dense(np.prod(encoder_shape))(z)
    decoder_input = Reshape(encoder_shape)(decoder_input)

    decoder = Conv2DTranspose(32, (3, 3), padding='same')(decoder_input)
    decoder = UpSampling2D((2, 2))(decoder)
    
    decoder = Conv2DTranspose(64, (3, 3), padding='same')(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    
    decoder = Conv2DTranspose(128, (3, 3), padding='same')(decoder)
    decoder = UpSampling2D((2, 2))(decoder)

    output_layer = Conv2D(3, (3, 3), padding='same', name="OUTPUT")(decoder)

    # 创建模型
    vae = Model(input_layer, output_layer)

    # 自定义损失函数
    reconstruction_loss = tf.keras.losses.binary_crossentropy(input_layer, output_layer) * np.prod(input_shape)
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae

traindir = "dataset/train/normal"
normal_dir = "dataset/val/normal"
anomaly_dir = "dataset/val/anomaly"

train_gen = data_generator(traindir, batch_size=16)
normal_gen = data_generator(normal_dir, batch_size=16)
anomaly_gen = data_generator(anomaly_dir, batch_size=16)

train_size = len([name for name in os.listdir(traindir) if name.endswith(('png', 'jpg'))])
normal_size = len([name for name in os.listdir(normal_dir) if name.endswith(('png', 'jpg'))])
anomaly_size = len([name for name in os.listdir(anomaly_dir) if name.endswith(('png', 'jpg'))])

print(f"训练数据集大小: {train_size}")
print(f"正常数据集大小: {normal_size}")
print(f"异常数据集大小: {anomaly_size}")

choice = input("您想训练一个新模型还是加载一个现有模型？(train/load): ").strip().lower()

if choice == 'train':
    vae = build_variational_autoencoder(input_shape=(448, 320, 3), latent_dim=512)
    vae.summary()
    
    history = vae.fit(train_gen,
                      steps_per_epoch=train_size // 16,
                      epochs=70,
                      validation_data=normal_gen,
                      validation_steps=normal_size // 16)
    
    vae.save("output/vae_model.h5")
    
    plt.plot(history.history['loss'])
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('轮次')
    plt.show()
else:
    vae = tf.keras.models.load_model('output/vae_model.h5', custom_objects={'sampling': sampling})

vae.summary()

normal_data_batch = next(normal_gen)[0]
reconstructed = vae.predict(normal_data_batch)
show_data(normal_data_batch, title="原始正常图片")
show_data(reconstructed, title="重建正常图片")

results = vae.evaluate(normal_gen, steps=normal_size // 8)
print("正常数据的损失和准确率", results)

anomaly_data_batch = next(anomaly_gen)[0]
reconstructed = vae.predict(anomaly_data_batch)
show_data(anomaly_data_batch, title="原始异常图片")
show_data(reconstructed, title="重建异常图片")

results = vae.evaluate(anomaly_gen, steps=anomaly_size // 8)
print("异常数据的损失和准确率", results)
