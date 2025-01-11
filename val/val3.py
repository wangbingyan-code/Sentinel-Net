import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# 打开文件选择器选择图片
def open_image():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename()
    return file_path

# 计算余弦距离
def cosine_distance(img1, img2):
    img1_flat = img1.flatten().reshape(1, -1)
    img2_flat = img2.flatten().reshape(1, -1)
    similarity = cosine_similarity(img1_flat, img2_flat)
    return 1 - similarity  # 余弦相似度转为余弦距离

# 计算欧式距离
def euclidean_distance(img1, img2):
    return euclidean(img1.flatten(), img2.flatten())

# 主函数：选择两张图片并计算距离
def main():
    print("请选择输入图像")
    img1_path = open_image()
    print(f"已选择输入图像: {img1_path}")

    print("请选择重建图像")
    img2_path = open_image()
    print(f"已选择重建图像: {img2_path}")

    # 打开图像并转换为 numpy 数组
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # 将图像 resize 为相同大小
    target_size = (320, 320)
    img1 = img1.resize(target_size)
    img2 = img2.resize(target_size)

    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # 计算余弦距离和欧式距离
    cos_dist = cosine_distance(img1_array, img2_array)
    euc_dist = euclidean_distance(img1_array, img2_array)

    print(f"余弦距离: {cos_dist[0][0]}")
    print(f"欧式距离: {euc_dist}")

if __name__ == "__main__":
    main()
