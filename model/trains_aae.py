import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
import pandas as pd
import time
import cv2
# 设置随机种子以保证实验可复现
np.random.seed(42)
tf.random.set_seed(42)

# 数据集加载函数
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

# 定义SSIM损失函数
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# 编码器
def build_encoder(input_shape=(448, 320, 3), latent_dim=128):
    inputs = Input(shape=input_shape, name="encoder_input")
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    latent = Dense(latent_dim, activation='linear', name="latent_vector")(x)
    return Model(inputs, latent, name="Encoder")

# 解码器
def build_decoder(output_shape=(448, 320, 3), latent_dim=128):
    latent_inputs = Input(shape=(latent_dim,), name="decoder_input")
    x = Dense(56 * 40 * 128, activation='relu')(latent_inputs)
    x = Reshape((56, 40, 128))(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', name="decoder_output")(x)
    return Model(latent_inputs, outputs, name="Decoder")

# 判别器
def build_discriminator(latent_dim=128):
    inputs = Input(shape=(latent_dim,), name="discriminator_input")
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid', name="discriminator_output")(x)
    return Model(inputs, outputs, name="Discriminator")

# 初始化模型
latent_dim = 128
input_shape = (448, 320, 3)
encoder = build_encoder(input_shape=input_shape, latent_dim=latent_dim)
decoder = build_decoder(output_shape=input_shape, latent_dim=latent_dim)
discriminator = build_discriminator(latent_dim=latent_dim)

# 编译判别器
discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss=BinaryCrossentropy(), metrics=['accuracy'])

# 构建对抗自编码器（AAE）
autoencoder_input = Input(shape=input_shape, name="autoencoder_input")
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)

discriminator.trainable = False
validity = discriminator(encoded)

aae = Model(autoencoder_input, [decoded, validity], name="AAE")
aae.compile(optimizer=Adam(learning_rate=0.0002),
            loss=[ssim_loss, BinaryCrossentropy()],
            loss_weights=[0.999, 0.001])

# 数据加载
train_normal_dir = "../dataset/train/normal"
val_normal_dir = "../dataset/val/normal"
val_anomaly_dir = "../dataset/val/anomaly"

train_dataset = create_dataset(train_normal_dir, batch_size=8)
val_normal_dataset = create_dataset(val_normal_dir, batch_size=8)
val_anomaly_dataset = create_dataset(val_anomaly_dir, batch_size=8)

# 训练
choice = input("您想训练一个新模型还是加载一个现有模型？(train/load): ").strip().lower()


if choice == 'train':
    epochs = 100
    loss_history = []  # 保存生成器（AAE）的损失
    discriminator_loss_history = []  # 保存判别器的损失

    for epoch in range(epochs):
        epoch_g_loss = []
        epoch_d_loss = []

        for batch in train_dataset:
            imgs, _ = batch
            latent_fake = encoder.predict(imgs)
            latent_real = np.random.normal(size=(imgs.shape[0], latent_dim))

            # 训练判别器
            d_loss_real = discriminator.train_on_batch(latent_real, np.ones((imgs.shape[0], 1)))
            d_loss_fake = discriminator.train_on_batch(latent_fake, np.zeros((imgs.shape[0], 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            epoch_d_loss.append(d_loss[0])

            # 训练生成器（编码器 + 解码器）
            g_loss = aae.train_on_batch(imgs, [imgs, np.ones((imgs.shape[0], 1))])
            epoch_g_loss.append(g_loss[0])

        # 记录每个epoch的平均损失
        loss_history.append(np.mean(epoch_g_loss))
        discriminator_loss_history.append(np.mean(epoch_d_loss))

        print(f"Epoch {epoch + 1}/{epochs} - D Loss: {np.mean(epoch_d_loss):.4f}, G Loss: {np.mean(epoch_g_loss):.4f}")

    # 保存模型
    os.makedirs('../output', exist_ok=True)
    aae.save("../output/model_aae.h5")

# 保存损失数据到 CSV 文件
    loss_data = {
        'epoch': range(1, epochs + 1),
        'generator_loss': loss_history,
        'discriminator_loss': discriminator_loss_history
    }
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv('../output/loss_data_aae.csv', index=False)
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, label='Generator Loss (AAE)', color='b')
    plt.plot(range(1, epochs + 1), discriminator_loss_history, label='Discriminator Loss', color='r')
    plt.title('Training Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('../output/loss_curve.png')
    plt.show()

elif choice == 'load':
    # 加载模型
    try:
        aae = load_model('../output/model_aae.h5', custom_objects={'ssim_loss': ssim_loss})
        print("模型加载成功！")
    except Exception as e:
        print("加载模型失败，请确认路径是否正确。", str(e))
        exit()

    # 提取自编码器
    encoder = aae.get_layer("Encoder")
    decoder = aae.get_layer("Decoder")

    autoencoder_input = Input(shape=(448, 320, 3), name="autoencoder_test_input")
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(autoencoder_input, decoded, name="Autoencoder_Test")
    print("提取出的自编码器已构建完成！")

    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# 打印模型信息
    autoencoder.summary()
    output_dir = '../output/reconstructed_img/'
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    # 验证集预测
    def evaluate_and_show(dataset, dataset_name):
        data_batch =  next(iter(dataset))[0]
        start_time = time.time()  # 开始计时
        reconstructed = autoencoder.predict(data_batch)
        end_time = time.time()  # 结束计时
        inference_time = end_time - start_time
        print(f"{dataset_name} 数据集推理耗时: {inference_time:.4f} 秒，平均每张图片耗时: {inference_time / len(data_batch):.4f} 秒")
        # 显示原始图片和重建图片
        plt.figure(figsize=(15, 5))
        for i in range(8):
    # 原始图片
            ax = plt.subplot(2, 8, i + 1)
            plt.imshow(data_batch[i])
            ax.set_title("Original", fontsize=10)
            ax.axis('off')

            # 重建图片
            ax = plt.subplot(2, 8, i + 9)
            plt.imshow(reconstructed[i])
            ax.set_title("Reconstructing", fontsize=10)
            ax.axis('off')

            # 保存重建图片到指定路径
            save_path = os.path.join(output_dir, f'reconstructed_{i}.jpg')  # 生成唯一文件名
            reconstructed_image = (reconstructed[i] * 255).astype('uint8')  # 如果 reconstructed 范围是 [0, 1]，需转换到 [0, 255]
            cv2.imwrite(save_path, reconstructed_image)  # 保存图片
            print(f"Saved reconstructed image {i} to {save_path}")

        # 显示拼接的图像
        plt.show()
        # 评估损失
        results = autoencoder.evaluate(dataset, steps=len(dataset), verbose=1)
        print(f"{dataset_name} 数据集的损失和准确率: {results}")

    # 对正常数据集和异常数据集进行预测和评估
    print("正常数据集预测：")
    evaluate_and_show(val_normal_dataset, "normal")

    print("异常数据集预测：")
    evaluate_and_show(val_anomaly_dataset, "anomaly")

else:
    print("无效输入，请输入 'train' 或 'load'")
# Total params: 74,055,427
# Trainable params: 74,055,427
# Non-trainable params: 0
# _________________________________________________________________
# 正常数据集预测：
# normal 数据集推理耗时: 0.9620 秒，平均每张图片耗时: 0.1202 秒
# 5/5 [==============================] - 1s 117ms/step - loss: 0.0015 - accuracy: 0.8585
# normal 数据集的损失和准确率: [0.001481350976973772, 0.858497679233551]
# 异常数据集预测：
# anomaly 数据集推理耗时: 0.0487 秒，平均每张图片耗时: 0.0061 秒
# 20/20 [==============================] - 0s 21ms/step - loss: 0.0137 - accuracy: 0.7856
# anomaly 数据集的损失和准确率: [0.013675335794687271, 0.7855940461158752]