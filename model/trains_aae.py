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
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Dataset loading function
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

# SSIM loss function
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Encoder
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

# Decoder
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

# Discriminator
def build_discriminator(latent_dim=128):
    inputs = Input(shape=(latent_dim,), name="discriminator_input")
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid', name="discriminator_output")(x)
    return Model(inputs, outputs, name="Discriminator")

# Initialize models
latent_dim = 128
input_shape = (448, 320, 3)
encoder = build_encoder(input_shape=input_shape, latent_dim=latent_dim)
decoder = build_decoder(output_shape=input_shape, latent_dim=latent_dim)
discriminator = build_discriminator(latent_dim=latent_dim)

# Compile discriminator
discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Build Adversarial Autoencoder (AAE)
autoencoder_input = Input(shape=input_shape, name="autoencoder_input")
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)

discriminator.trainable = False
validity = discriminator(encoded)

aae = Model(autoencoder_input, [decoded, validity], name="AAE")
aae.compile(optimizer=Adam(learning_rate=0.0002),
            loss=[ssim_loss, BinaryCrossentropy()],
            loss_weights=[0.999, 0.001])

# Data loading
train_normal_dir = "../dataset/train/normal"
val_normal_dir = "../dataset/val/normal"
val_anomaly_dir = "../dataset/val/anomaly"

train_dataset = create_dataset(train_normal_dir, batch_size=8)
val_normal_dataset = create_dataset(val_normal_dir, batch_size=8)
val_anomaly_dataset = create_dataset(val_anomaly_dir, batch_size=8)

# Training
choice = input("Do you want to train a new model or load an existing one? (train/load): ").strip().lower()

if choice == 'train':
    epochs = 100
    loss_history = []  # Save generator (AAE) loss
    discriminator_loss_history = []  # Save discriminator loss

    for epoch in range(epochs):
        epoch_g_loss = []
        epoch_d_loss = []

        for batch in train_dataset:
            imgs, _ = batch
            latent_fake = encoder.predict(imgs)
            latent_real = np.random.normal(size=(imgs.shape[0], latent_dim))

            # Train discriminator
            d_loss_real = discriminator.train_on_batch(latent_real, np.ones((imgs.shape[0], 1)))
            d_loss_fake = discriminator.train_on_batch(latent_fake, np.zeros((imgs.shape[0], 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            epoch_d_loss.append(d_loss[0])

            # Train generator (encoder + decoder)
            g_loss = aae.train_on_batch(imgs, [imgs, np.ones((imgs.shape[0], 1))])
            epoch_g_loss.append(g_loss[0])

        # Record average loss per epoch
        loss_history.append(np.mean(epoch_g_loss))
        discriminator_loss_history.append(np.mean(epoch_d_loss))

        print(f"Epoch {epoch + 1}/{epochs} - D Loss: {np.mean(epoch_d_loss):.4f}, G Loss: {np.mean(epoch_g_loss):.4f}")

    # Save model
    os.makedirs('../output', exist_ok=True)
    aae.save("../output/model_aae.h5")

    # Save loss data to CSV file
    loss_data = {
        'epoch': range(1, epochs + 1),
        'generator_loss': loss_history,
        'discriminator_loss': discriminator_loss_history
    }
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv('../output/loss_data_aae.csv', index=False)
    # Plot loss curve
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
    # Load model
    try:
        aae = load_model('../output/model_aae.h5', custom_objects={'ssim_loss': ssim_loss})
        print("Model loaded successfully!")
    except Exception as e:
        print("Failed to load model, please check the path.", str(e))
        exit()

    # Extract autoencoder
    encoder = aae.get_layer("Encoder")
    decoder = aae.get_layer("Decoder")

    autoencoder_input = Input(shape=(448, 320, 3), name="autoencoder_test_input")
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(autoencoder_input, decoded, name="Autoencoder_Test")
    print("Extracted autoencoder has been built!")

    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # Print model summary
    autoencoder.summary()
    output_dir = '../output/reconstructed_img/'
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    # Validation set prediction
    def evaluate_and_show(dataset, dataset_name):
        data_batch =  next(iter(dataset))[0]
        start_time = time.time()  # Start timing
        reconstructed = autoencoder.predict(data_batch)
        end_time = time.time()  # End timing
        inference_time = end_time - start_time
        print(f"{dataset_name} dataset inference time: {inference_time:.4f} s, average per image: {inference_time / len(data_batch):.4f} s")
        # Show original and reconstructed images
        plt.figure(figsize=(15, 5))
        for i in range(8):
            # Original image
            ax = plt.subplot(2, 8, i + 1)
            plt.imshow(data_batch[i])
            ax.set_title("Original", fontsize=10)
            ax.axis('off')

            # Reconstructed image
            ax = plt.subplot(2, 8, i + 9)
            plt.imshow(reconstructed[i])
            ax.set_title("Reconstructed", fontsize=10)
            ax.axis('off')

            # Save reconstructed image to specified path
            save_path = os.path.join(output_dir, f'reconstructed_{i}.jpg')  # Generate unique filename
            reconstructed_image = (reconstructed[i] * 255).astype('uint8')  # If reconstructed is in [0, 1], convert to [0, 255]
            cv2.imwrite(save_path, reconstructed_image)  # Save image
            print(f"Saved reconstructed image {i} to {save_path}")

        # Show concatenated images
        plt.show()
        # Evaluate loss
        results = autoencoder.evaluate(dataset, steps=len(dataset), verbose=1)
        print(f"{dataset_name} dataset loss and accuracy: {results}")

    # Predict and evaluate on normal and anomaly datasets
    print("Normal dataset prediction:")
    evaluate_and_show(val_normal_dataset, "normal")

    print("Anomaly dataset prediction:")
    evaluate_and_show(val_anomaly_dataset, "anomaly")

else:
    print("Invalid input, please enter 'train' or 'load'")
# Total params: 74,055,427
# Trainable params: 74,055,427
# Non-trainable params: 0
# _________________________________________________________________
# Normal dataset prediction:
# normal dataset inference time: 0.9620 s, average per image: 0.1202 s
# 5/5 [==============================] - 1s 117ms/step - loss: 0.0015 - accuracy: 0.8585
# normal dataset loss and accuracy: [0.001481350976973772, 0.858497679233551]
# Anomaly dataset prediction:
# anomaly dataset inference time: 0.0487 s, average per image: 0.0061 s
# 20/20 [==============================] - 0s 21ms/step - loss: 0.0137 - accuracy: 0.7856
# anomaly dataset loss and accuracy: