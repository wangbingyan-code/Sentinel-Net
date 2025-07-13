import os
from PIL import Image
import matplotlib.pyplot as plt

input_folder = "dataset/data"
output_folder = "dataset/data_croped"

os.makedirs(output_folder, exist_ok=True)

# Define crop region (left, top, right, bottom)
left = 700
top = 80
right = 1300
bottom = 800

count = 0
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png")):
        input_image_path = os.path.join(input_folder, filename)
        try:
            image = Image.open(input_image_path)
            cropped_image = image.crop((left, top, right, bottom))
            output_image_path = os.path.join(output_folder, filename)
            cropped_image.save(output_image_path)
            print(f"Cropped and saved image: {output_image_path}")
            count += 1
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

print(f"Total {count} images processed and saved.")