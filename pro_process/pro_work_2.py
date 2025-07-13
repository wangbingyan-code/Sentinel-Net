import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
def open_image():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename()
    return file_path

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
        return output_image, diff, thresh
    else:
        return image02_resized, diff, thresh

if __name__ == "__main__":
    image01_path = 'dataset/reconstructed_image/reconstructed.jpg'
    image02_path = open_image()
    image01 = cv2.imread(image01_path)
    image02 = cv2.imread(image02_path)

    output_image, diff, thresh = compute_difference_and_draw_boxes(image01, image02)
    # cv2.imshow('Difference', diff)  
    # cv2.imshow('Threshold', thresh)  
    cv2.imshow('Output on Image 2', output_image)  
    cv2.imwrite('result_image.jpg',output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()