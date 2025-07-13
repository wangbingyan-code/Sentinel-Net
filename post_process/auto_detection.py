import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import tensorflow as tf
from PIL import Image

# Model file path
model_path = "output/model.h5"

def ssim_loss(y_true, y_pred):
    # Loss function
    ssim_loss_value = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    l2_loss_value = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)))
    return ssim_loss_value + 0.25 * l2_loss_value

def preprocess_image(input_image, target_size=(320, 448)):
    img_array = cv2.resize(input_image, target_size)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def extract_frames(input_video, fps=1):
    # Extract one frame per second from the video
    cap = cv2.VideoCapture(input_video)
    frames = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate // fps)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

def compute_reconstruction_error(image_data):
    results = autoencoder.evaluate(image_data, image_data, batch_size=2)
    formatted_result = f"{results[0]:.3f}"
    return formatted_result

def open_file():
    # Open video file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

# Define crop region (left, top, right, bottom)
left = 700
top = 80
right = 1300
bottom = 800

def crop(input_image):
    # Crop the image to the specified region
    pil_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    cropped_image = pil_image.crop((left, top, right, bottom))
    cropped_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
    return cropped_image

def compute_difference_and_draw_boxes(image01, image02):
    image02_resized = cv2.resize(image02, (image01.shape[1], image01.shape[0]))
    gray1 = cv2.cvtColor(image01, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image02_resized, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        output_image = image02_resized.copy()
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output_image, "Anomaly", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return output_image
    else:
        return image02_resized

def process_video(input_video, output_video, model, fps=1):
    frames = extract_frames(input_video, fps)
    height, width, _ = frames[0].shape

    # Define output video codec and file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame in frames:
        cropped_image = crop(frame)
        preprocessed_image = preprocess_image(cropped_image)
        # Compute reconstruction loss
        reconstruct_loss = compute_reconstruction_error(preprocessed_image)
        reconstruct_loss = float(reconstruct_loss)
        if reconstruct_loss > 0.35:  # If reconstruction loss is greater than threshold, consider as anomaly
            reconstructed = model.predict(preprocessed_image)
            reconstructed = np.squeeze(reconstructed, axis=0)
            reconstructed = np.clip(reconstructed, 0, 1)
            reconstructed = (reconstructed * 255).astype(np.uint8)
            # Draw anomaly box
            output_frame = compute_difference_and_draw_boxes(reconstructed, cropped_image)
            cv2.imshow("output_frame", output_frame)
        else:
            output_frame = cropped_image
            cv2.imshow("output_frame_00", output_frame)
        # Write processed frame to output video
        out.write(output_frame)
    out.release()

if __name__ == "__main__":
    input_video_path = filedialog.askopenfilename()  # Select video file
    output_video_path = "dataset/video/detection_output.mp4"
    autoencoder = load_model(model_path, custom_objects={'ssim_loss': ssim_loss})  # Load model
    process_video(input_video_path, output_video_path, autoencoder, fps=1)  # Process video and save
    # Play the generated video
    cap = cv2.VideoCapture(output_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Anomaly Detection Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break