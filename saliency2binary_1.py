import cv2
import numpy as np
import os

# Define tha path of input folder and output folder
input_folder = 'picture/saliency/'
output_folder = 'picture/binary/'

# Ensuring the exist of output folder
os.makedirs(output_folder, exist_ok= True)

# Batch process the image in folder
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # thresholding
    threshold_value = 200
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Opening-Closing operation
    kernel = np.ones((7, 7), np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

    # Save the processed image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path,closed_image)






