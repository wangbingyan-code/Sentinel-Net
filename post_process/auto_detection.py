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

# modle file path
model_path = "output/model.h5"

def ssim_loss(y_true, y_pred):
    # loss function
    ssim_loss_value = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    # l2_loss_value = tf.reduce_mean(tf.square(y_true - y_pred))  
    l2_loss_value = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)))
    return ssim_loss_value + 0.25*l2_loss_value


def preprocess_image(input_image, target_size=(320, 448)):
    img_array = cv2.resize(input_image, target_size)
    img_array = img_array.astype('float32') / 255.0  # 归一化到 [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # 增加批处理维度
    return img_array

def extract_frames(input_video, fps=1):
    # 实现每一秒提取出一帧画面
    cap = cv2.VideoCapture(input_video)
    frames = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    interval = int(frame_rate // fps)  # 每隔多少帧提取一帧
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

def compute_reconstruction_error(image_date):
    results = autoencoder.evaluate(image_date, image_date, batch_size=2)
    # print(results)
    formatted_result = f"{results[0]:.3f}"  
    return formatted_result

def open_file():
    # open video file
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename()
    return file_path

# 定义裁剪区域 (左, 上, 右, 下)
left = 700
top = 80
right = 1300
bottom = 800

def crop(input_image):
    # 将图片裁剪到指定大小
    pil_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))  # 转换为PIL图像
    cropped_image = pil_image.crop((left, top, right, bottom))  # 裁剪
    cropped_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)  # 转换回OpenCV格式
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
    

def process_video(input_video, output_video,model, fps=1):
    frames = extract_frames(input_video, fps)  # 提取视频帧
    height, width, _ = frames[0].shape  # 获取帧的尺寸
    
    # 定义输出视频编解码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for frame in frames:
        cropped_image = crop(frame)  # 裁剪帧
        preprocessed_image = preprocess_image(cropped_image)  # 预处理帧
        # 计算重建损失
        reconstruct_loss = compute_reconstruction_error(preprocessed_image)
        reconstruct_loss = float(reconstruct_loss)
        if reconstruct_loss > 0.35:  # 重建损失大于阈值，视为异常
            reconstructed = model.predict(preprocessed_image)
            reconstructed = np.squeeze(reconstructed, axis=0)
            reconstructed = np.clip(reconstructed, 0, 1)  # 数据范围[0, 1]
            reconstructed = (reconstructed * 255).astype(np.uint8)
            # 绘制异常框
            
            output_frame = compute_difference_and_draw_boxes(reconstructed, cropped_image)
            cv2.imshow("output_frame",output_frame)
        else:
            output_frame = cropped_image  # 如果没有异常，使用原始裁剪后的帧
            cv2.imshow("output_frame_00",output_frame)
        # 将处理后的帧写入输出视频
        out.write(output_frame)
    out.release()


if __name__ =="__main__":
    # video_path = open_file()   #选中视频文件
    # frames = extract_frames(video_path,1) #提取视频的帧图像
    # input_image_01 = crop(frames) #裁剪图像
    # input_image_02 = preprocess_image(input_image_01)#预处理
    # autoencoder = load_model(model_path, custom_objects={'ssim_loss': ssim_loss})#加载模型文件
    # reconstructed = autoencoder.predict(input_image_02)
    # reconstructed = np.squeeze(reconstructed, axis=0)
    # reconstructed = np.clip(reconstructed, 0, 1)  # 如果数据范围在 [0, 1]
    # reconstructed = (reconstructed * 255).astype(np.uint8)
    # #reconstructed 是重建图像
    # output_image = compute_difference_and_draw_boxes(reconstructed,input_image_01)
    # video_path = open_file()  # 选中视频文件
    # frames = extract_frames(video_path, fps=1)  # 每秒提取一帧图像
    # for frame in frames:
    #     input_image_01 = crop(frame)  # 裁剪图像
    #     input_image_02 = preprocess_image(input_image_01)  # 预处理
    #     autoencoder = load_model(model_path, custom_objects={'ssim_loss': ssim_loss})  # 加载模型文件
    #     reconstruct_loss = compute_reconstruction_error(input_image_02)#计算重建损失
    #     if reconstruct_loss >0.35 : #重建损失大于0.35认为输入的视频帧是异常帧
    #         reconstructed = autoencoder.predict(input_image_02)  # 使用模型重建图像
    #         reconstructed = np.squeeze(reconstructed, axis=0)
    #         reconstructed = np.clip(reconstructed, 0, 1)  # 如果数据范围在 [0, 1]
    #         reconstructed = (reconstructed * 255).astype(np.uint8)
            
    #         # 计算差异并绘制异常框
    #         output_image = compute_difference_and_draw_boxes(reconstructed, input_image_01)
            
    #         # 显示或保存结果
    #         plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    #         plt.show()
    input_video_path = filedialog.askopenfilename()  # 选择视频文件
    output_video_path = "dataset/video/detection_output.mp4"
    autoencoder = load_model(model_path, custom_objects={'ssim_loss': ssim_loss})  # 加载模型
    process_video(input_video_path, output_video_path, autoencoder, fps=1)  # 处理视频并保存
    # 播放生成的视频
    cap = cv2.VideoCapture(output_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Anomaly Detection Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    