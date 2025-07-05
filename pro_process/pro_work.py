import cv2
import os
import matplotlib.pyplot as plt

def load_images(image_path1, image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1.shape != img2.shape:
        raise ValueError("两帧图像的尺寸不一致")
    
    return img1, img2


def compute_difference(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return diff

def process_images(input_folder, output_folder):
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(len(image_files) - 1):
        img_path1 = os.path.join(input_folder, image_files[i])
        img_path2 = os.path.join(input_folder, image_files[i+1])
        
        img1, img2 = load_images(img_path1, img_path2)
        diff_img = compute_difference(img1, img2)
        diff_image_name = f"diff_{i+1:04d}.jpg"
        cv2.imwrite(os.path.join(output_folder, diff_image_name), diff_img)
        print(f"保存差分图像: {diff_image_name}")

if __name__ == "__main__":
    input_folder = 'dataset/train/normal'
    output_folder = 'dataset/diff_image/normal'

    process_images(input_folder, output_folder)
