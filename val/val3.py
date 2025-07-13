import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

def open_image():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename()
    return file_path

def cosine_distance(img1, img2):
    img1_flat = img1.flatten().reshape(1, -1)
    img2_flat = img2.flatten().reshape(1, -1)
    similarity = cosine_similarity(img1_flat, img2_flat)
    return 1 - similarity  

def euclidean_distance(img1, img2):
    return euclidean(img1.flatten(), img2.flatten())

def main():
    print("Please select the input image.")
    img1_path = open_image()
    print(f"Selected input image: {img1_path}")

    print("Please select the reconstructed image.")
    img2_path = open_image()
    print(f"Selected reconstructed image: {img2_path}")

    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    target_size = (320, 320)
    img1 = img1.resize(target_size)
    img2 = img2.resize(target_size)

    img1_array = np.array(img1)
    img2_array = np.array(img2)

    cos_dist = cosine_distance(img1_array, img2_array)
    euc_dist = euclidean_distance(img1_array, img2_array)

    print(f"Cosine distance: {cos_dist[0][0]}")
    print(f"Euclidean distance: {euc_dist}")

if __name__ == "__main__":  
    main()