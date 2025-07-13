import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError
from keras.models import load_model

class Sampling(Layer):
    def __init__(self):
        super(Sampling, self).__init__()

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def ssim_loss(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

def calculate_psnr(y_true, y_pred, max_val=1.0):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    psnr = 10.0 * tf.math.log((max_val ** 2) / (mse + 1e-10)) / tf.math.log(10.0)
    return psnr

def calculate_reconstruction_error_with_predict(autoencoder, dataset, batch_size=8):
    losses = []
    psnrs = []
    for img in dataset:
        img = np.expand_dims(img, axis=0)
        reconstructed_image = autoencoder.predict(img, batch_size=batch_size)
        loss, _ = autoencoder.evaluate(img, img, verbose=0)
        psnr = calculate_psnr(img, reconstructed_image).numpy()
        losses.append(loss)
        psnrs.append(psnr)
        print(f"Reconstruction error: {loss}, PSNR: {psnr}")
    return np.array(losses), np.array(psnrs)

# Image preprocessing function
def preprocess_image(image_path, target_size=(448, 320)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img).astype('float32') / 255.0
    return img_array

# Load dataset
def create_dataset(directory, target_size=(448, 320)):
    filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]
    dataset = np.array([preprocess_image(f, target_size) for f in filepaths])
    return dataset

# Plot ROC curve
def plot_roc_curve(fpr, tpr, thresholds, roc_auc, optimal_threshold):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()

    # Mark the optimal threshold point
    optimal_idx = np.argmax(tpr - fpr)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.legend(loc='best')
    plt.show()

def main():
    # Path configuration
    model_path = "../output/vae_model.h5"
    normal_dir = "../dataset/val/normal"
    anomaly_dir = "../dataset/val/anomaly"
    
    # Load model and data
    autoencoder = load_model(
        model_path,
        custom_objects={'ssim_loss': ssim_loss, 'sampling': Sampling}
    )

    normal_dataset = create_dataset(normal_dir)
    anomaly_dataset = create_dataset(anomaly_dir)
    
    # Calculate reconstruction error and PSNR for normal and anomaly datasets
    normal_errors, normal_psnrs = calculate_reconstruction_error_with_predict(autoencoder, normal_dataset)
    average_normal_psnr = np.mean(normal_psnrs)

    print("Normal image errors end########################################################")
    anomaly_errors, anomaly_psnrs = calculate_reconstruction_error_with_predict(autoencoder, anomaly_dataset)
    
    # Merge data
    all_errors = np.concatenate([normal_errors, anomaly_errors], axis=0).flatten()
    labels = np.concatenate([np.zeros_like(normal_errors), np.ones_like(anomaly_errors)], axis=0).flatten()
    fpr, tpr, thresholds = roc_curve(labels, all_errors)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)
    plt.title('VAE ROC Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid()

    # Mark optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.legend(loc='best')
    plt.show()

    # Output threshold, TPR, FPR
    for i, threshold in enumerate(thresholds):
        print(f"Threshold: {threshold:.4f}, TPR: {tpr[i]:.4f}, FPR: {fpr[i]:.4f}")

if __name__ == "__main__":
    main()