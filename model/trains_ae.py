import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, Dense, Lambda, Dropout,Reshape
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.font_manager as fm
import pandas as pd
from tensorflow.keras.losses import MeanSquaredError
import time
def ssim_loss(y_true, y_pred):
   return MeanSquaredError()(y_true, y_pred)

def set_chinese_font():
    font_path = "C:/Windows/Fonts/simhei.ttf"  
    if os.path.exists(font_path):
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
    else:
        print("Font file not found, please adjust the path.")

set_chinese_font()

def create_dataset(directory, batch_size=32, target_size=(448, 320)):
    filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]

    def load_and_preprocess_image(filename):
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = img / 255.0  # Normalize to [0, 1]
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

def build_autoencoder(input_shape=(448, 320, 3), activation='relu'):
    input_layer = Input(shape=input_shape, name="INPUT")
    encoder = layers.GaussianNoise(stddev=0.2)(input_layer)
    # print("Input layer shape:", input_layer.shape)
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
    
    
    
    # encoder = Conv2D(256, (3, 3), padding='same')(encoder)
    # encoder = activation_fn(encoder)
    # encoder = Dropout(0.3)(encoder)
    # encoder = MaxPooling2D((2, 2))(encoder)
    
    encoder = Conv2D(128, (5, 5), padding='same')(encoder)
    encoder = activation_fn(encoder)
    encoder = Dropout(0.3)(encoder)
    encoder = MaxPooling2D((2, 2))(encoder)
    
    encoder = Conv2D(64, (3, 3), padding='same')(encoder)
    encoder = activation_fn(encoder)
    encoder = Dropout(0.3)(encoder)
    encoder = MaxPooling2D((2, 2))(encoder)
    
    encoder = Conv2D(32, (5, 5), padding='same')(encoder)
    encoder = activation_fn(encoder)
    encoder = Dropout(0.3)(encoder)
    encoder = MaxPooling2D((2, 2))(encoder)
    
    encoder = Conv2D(16, (3, 3), padding='same')(encoder)
    encoder = activation_fn(encoder)
    encoder = Dropout(0.3)(encoder)
    encoded = MaxPooling2D((2, 2))(encoder)
    # ###################################
    encoder_shape = tf.keras.backend.int_shape(encoded)[1:]  
    encoded = Flatten()(encoded)
    encoded = Dense(512)(encoded)
    encoded = activation_fn(encoded)
    # print("Encoder output shape:", encoded.shape)
    ####################################################
    decoder = Dense(np.prod(encoder_shape))(encoded)  
    decoder = activation_fn(decoder)
    decoder = Reshape(encoder_shape)(decoder)
    
    # Decoder
    decoder = Conv2DTranspose(16, (3, 3), padding='same')(decoder)
    decoder = activation_fn(decoder)
    decoder = Dropout(0.3)(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    
    decoder = Conv2DTranspose(32, (5, 5), padding='same')(decoder)
    decoder = activation_fn(decoder)
    decoder = Dropout(0.3)(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    
    decoder = Conv2DTranspose(64, (3, 3), padding='same')(decoder)
    decoder = activation_fn(decoder)
    decoder = Dropout(0.3)(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    
    decoder = Conv2DTranspose(128, (5, 5), padding='same')(decoder)
    decoder = activation_fn(decoder)
    decoder = Dropout(0.3)(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    
    # decoder = Conv2DTranspose(256, (3, 3), padding='same')(decoder)
    # decoder = activation_fn(decoder)
    # decoder = Dropout(0.3)(decoder)
    # decoder = UpSampling2D((2, 2))(decoder)
    
    output_layer = Conv2D(3, (3, 3), padding='same', name="OUTPUT")(decoder)
    # print("Decoder output shape:", output_layer.shape) 
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(loss=ssim_loss, optimizer='adam', metrics=['accuracy'])
    return autoencoder
traindir = "../dataset/train/normal"
normal_dir = "../dataset/val/normal"
anomaly_dir = "../dataset/val/anomaly"

train_dataset = create_dataset(traindir, batch_size=8)
normal_dataset = create_dataset(normal_dir, batch_size=8)
anomaly_dataset = create_dataset(anomaly_dir, batch_size=8)

train_size = len([name for name in os.listdir(traindir) if name.endswith(('png', 'jpg'))])
normal_size = len([name for name in os.listdir(normal_dir) if name.endswith(('png', 'jpg'))])
anomaly_size = len([name for name in os.listdir(anomaly_dir) if name.endswith(('png', 'jpg'))])

print(f"Train dataset size: {train_size}")
print(f"Normal dataset size: {normal_size}")
print(f"Anomaly dataset size: {anomaly_size}")

choice = input("Do you want to train a new model or load an existing one? (train/load): ").strip().lower()

if choice == 'train':
    activation_choice = input("Choose activation function (relu/leaky_relu/elu/prelu/swish): ").strip().lower()
    autoencoder = build_autoencoder(input_shape=(448, 320, 3), activation=activation_choice)
    autoencoder.summary()

    history =autoencoder.fit(train_dataset,
                                epochs=100,
                                validation_data=normal_dataset)
    
    autoencoder.save("../output/model.h5")
    
    loss_data = history.history['loss']
    df = pd.DataFrame({'epoch': range(1, len(loss_data) + 1), 'loss': loss_data})
    df.to_csv('../output/loss_data_ae.csv', index=False)
    
    plt.plot(history.history['loss'], label='Loss')  
    plt.title('Model Loss', fontsize=16)  
    plt.ylabel('Loss', fontsize=14)  
    plt.xlabel('Epochs', fontsize=14)  
    plt.legend(fontsize=12)  
    plt.grid() 
    plt.show()

else:
    autoencoder = load_model('../output/model.h5', custom_objects={'ssim_loss': ssim_loss})

autoencoder.summary()

normal_data_batch = next(iter(normal_dataset))[0]
start_time = time.time()  
reconstructed = autoencoder.predict(normal_data_batch)
end_time = time.time()  
inference_time = end_time - start_time
print(f"normal_dataset inference time: {inference_time:.4f} s, average per image: {inference_time / normal_size:.4f} s")

show_data(normal_data_batch, title="Original normal images")
show_data(reconstructed, title="Reconstructed normal images")
results = autoencoder.evaluate(normal_dataset, steps=normal_size // 8)
print("Normal data loss and accuracy", results)

anomaly_data_batch = next(iter(anomaly_dataset))[0]
start_time = time.time()
reconstructed = autoencoder.predict(anomaly_data_batch)
end_time = time.time()  
inference_time = end_time - start_time
print(f"anomaly_dataset inference time: {inference_time:.4f} s, average per image: {inference_time / anomaly_size:.4f} s")

show_data(anomaly_data_batch, title="Original anomaly images")
show_data(reconstructed, title="Reconstructed anomaly images")

results = autoencoder.evaluate(anomaly_dataset, steps=anomaly_size // 8)
print("Anomaly data loss and accuracy", results)

# Total params: 9,565,923
# Trainable params: 9,565,923
# Non-trainable params: 0
# normal_dataset inference time: 1.4152 s, average per image: 0.0404 s
# 4/4 [==============================] - 1s 54ms/step - loss: 0.0021 - accuracy: 0.8615
# Normal data loss and accuracy [0.0021277335472404957, 0.8615419864654541]
# anomaly_dataset inference time: 0.0769 s, average per image: 0.0005 s
# 19/19 [==============================] - 1s 51ms/step - loss: 0.0132 - accuracy: 0.7884
# Anomaly data loss and accuracy [0.013158895075321198,