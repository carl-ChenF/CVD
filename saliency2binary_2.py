import cv2
import numpy as np
import os

# 定义输入和输出文件夹路径
input_folder = 'picture/saliency/'
output_folder = 'picture/binary/'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 批量处理文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 构建完整的文件路径
        input_path = os.path.join(input_folder, filename)
        
        # 读取二值图像
        saliency_map = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # 二值化显著性图，提取亮区
        _, binary_image = cv2.threshold(saliency_map, 200, 255, cv2.THRESH_BINARY)

        # 找到所有轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 计算总像素数
        total_pixels = binary_image.size

        # 计算阈值（0.1%）
        threshold_pixels = total_pixels * 0.0001

        # 遍历每个轮廓
        for contour in contours:
            # 计算轮廓内的像素数
            mask = np.zeros_like(binary_image)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            white_pixels = cv2.countNonZero(mask)

            # 判断像素数量是否少于阈值
            if white_pixels < threshold_pixels:
                # 将该图案设置为黑色
                binary_image[mask == 255] = 0

        # 对处理后的图像进行膨胀操作
        kernel = np.ones((51, 51), np.uint8)  # 定义膨胀核，大小为 5x5，可根据需要调整
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

        # 保存处理后的图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, binary_image)

        print(f'Processed and saved: {output_path}')
