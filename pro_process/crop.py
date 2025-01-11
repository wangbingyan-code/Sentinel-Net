import os
from PIL import Image
import matplotlib.pyplot as plt


input_folder = "dataset/data"
output_folder = "dataset/data_croped"

os.makedirs(output_folder, exist_ok=True)

# 定义裁剪区域 (左, 上, 右, 下)
left = 700
top = 80
right = 1300
bottom = 800


for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
    
        input_image_path = os.path.join(input_folder, filename)
        image = Image.open(input_image_path)

      
        cropped_image = image.crop((left, top, right, bottom))

       
        output_image_path = os.path.join(output_folder, filename)

        
        cropped_image.save(output_image_path)
        print(f"已裁剪并保存图像: {output_image_path}")

    
