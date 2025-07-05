import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError


# 自定义损失函数
def ssim_loss(y_true, y_pred):
    ssim_loss_value = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    l2_loss_value = tf.reduce_mean(tf.square(y_true - y_pred))  
    # l1_loss_value = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.5*ssim_loss_value + 0.5*l2_loss_value

def calculate_psnr(y_true, y_pred, max_val=1.0):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))  # 计算均方误差
    psnr = 10.0 * tf.math.log((max_val ** 2) / (mse + 1e-10)) / tf.math.log(10.0)  # 计算 PSNR
    return psnr

def calculate_reconstruction_error_with_predict(autoencoder, dataset, batch_size=8):
    losses = []
    psnrs = []
    for img in dataset:
        img = np.expand_dims(img, axis=0)  # 扩展维度以匹配模型输入
        reconstructed_image = autoencoder.predict(img, batch_size=batch_size)  # 推理出单张重建图像
        loss, _ = autoencoder.evaluate(img, img, verbose=0)  # 单张图像的重建误差
        psnr = calculate_psnr(img, reconstructed_image).numpy()  # 单张图像的 PSNR
        losses.append(loss)
        psnrs.append(psnr)
        print(f"重建误差：{loss}, PSNR: {psnr}")
    return np.array(losses), np.array(psnrs)

# 图像预处理函数
def preprocess_image(image_path, target_size=(448, 320)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img).astype('float32') / 255.0
    return img_array


# 加载数据集
def create_dataset(directory, target_size=(448, 320)):
    filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]
    dataset = np.array([preprocess_image(f, target_size) for f in filepaths])
    return dataset


# 绘制 ROC 曲线
def plot_roc_curve(fpr, tpr, thresholds, roc_auc, optimal_threshold):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()

    # 标注最佳阈值点
    optimal_idx = np.argmax(tpr - fpr)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.legend(loc='best')
    plt.show()


# 主函数
def main():
    # 配置路径
    model_path = "../output/20241211_180919/model_attention.h5"
    normal_dir = "../dataset/val/normal"
    anomaly_dir = "../dataset/val/anomaly"
    
    # 加载模型和数据
    autoencoder = load_model(model_path, custom_objects={'ssim_loss': ssim_loss})
    normal_dataset = create_dataset(normal_dir)
    anomaly_dataset = create_dataset(anomaly_dir)
    
    # 计算正常与异常数据集的重建误差和 PSNR
    normal_errors, normal_psnrs = calculate_reconstruction_error_with_predict(autoencoder, normal_dataset)
    average_normal_psnr = np.mean(normal_psnrs)

    print("normal image errors end########################################################")
    anomaly_errors, anomaly_psnrs = calculate_reconstruction_error_with_predict(autoencoder, anomaly_dataset)
    
    # 合并数据
    all_errors = np.concatenate([normal_errors, anomaly_errors], axis=0).flatten()  # 确保为一维数组
    # all_psnrs = np.concatenate([normal_psnrs, anomaly_psnrs], axis=0).flatten()  # 确保为一维数组
    
    
    labels = np.concatenate([np.zeros_like(normal_errors), np.ones_like(anomaly_errors)], axis=0).flatten()  # 确保为一维数组
    fpr, tpr, thresholds = roc_curve(labels, all_errors)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)
    plt.title('Sentinel ROC Curve',fontsize = 16)
    plt.legend(loc='lower right',fontsize=16)
    plt.grid()

    # optimal_threshold = 0.005  # 示例阈值
    # optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    # plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Threshold = {optimal_threshold:.4f}')
    plt.legend(loc='best')
    plt.show()

    # 输出阈值与 TPR/FPR
    for i, threshold in enumerate(thresholds):
        print(f"阈值: {threshold:.4f}, TPR: {tpr[i]:.4f}, FPR: {fpr[i]:.4f}")

    # 打印最佳阈值
    # print(f"\n示例阈值: {optimal_threshold:.4f}, 对应 TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f},Normal PSNR 平均值: {average_normal_psnr:.2f}")

    
if __name__ == "__main__":
    main()
