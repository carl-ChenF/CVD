import cv2
import numpy as np

def extract_and_crop_signal(original_image, saliency_map):
    # 确保显著性图是灰度格式
    if len(saliency_map.shape) == 3:
        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
    
    # 二值化显著性图，提取亮区
    _, binary_mask = cv2.threshold(saliency_map, 200, 255, cv2.THRESH_BINARY)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最大的轮廓（信号灯区域）
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 稍微扩大区域以确保完全包含信号灯
        padding = 100
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(original_image.shape[1] - x, w + 2*padding)
        h = min(original_image.shape[0] - y, h + 2*padding)
        
        # 直接裁剪出这个区域
        cropped_signal = original_image[y:y+h, x:x+w]
        
        return cropped_signal
    
    return None

# 加载图片
original_image = cv2.imread('test_01.jpg')
saliency_map = cv2.imread('saliency_map.png')

# 检查图片是否成功加载
if original_image is None or saliency_map is None:
    print("Error: Could not load one or both images")
    exit()

# 处理图片
result = extract_and_crop_signal(original_image, saliency_map)

if result is not None:
    cv2.imwrite('cropped_signal.jpg', result)

else:
    print("Error: Could not find signal light region")