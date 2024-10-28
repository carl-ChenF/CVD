import cv2
import os

def calculate_saliency(original_image_path, cvd_image_path):
    # Input the original image and cvd-simulated image
    original_img = cv2.imread(original_image_path)
    cvd_img = cv2.imread(cvd_image_path)
    
    # Convert the image to RGB formulate
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    cvd_img = cv2.cvtColor(cvd_img, cv2.COLOR_BGR2RGB)

    # Calculate the saliency map
    saliency_map = cv2.absdiff(original_img, cvd_img)

    # Convert the saliency map to gray imge
    saliency_map_gray = cv2.cvtColor(saliency_map, cv2.COLOR_RGB2BGR)

    # Normalize the saliency map
    saliency_map_normalized = cv2.normalize(saliency_map_gray, None, 0, 255, cv2.NORM_MINMAX)

    return saliency_map_normalized


def batch_process_saliency(original_image_dir, cvd_image_dir, output_dir):

    for filename in os.listdir(original_image_dir):
        original_image_path = os.path.join(original_image_dir, filename)
        cvd_image_path = os.path.join(cvd_image_dir, filename)

        # Load the image
        original_image = cv2.imread(original_image_path)
        cvd_image = cv2.imread(cvd_image_path)

        if original_image is None or cvd_image is None:
            print(f'Cannot find image at {original_image_path} or {cvd_image_path}')
            continue

        # Convert image to RGB format
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        cvd_image = cv2.cvtColor(cvd_image, cv2.COLOR_BGR2RGB)

        # Calculate the saliency map
        saliency_map = cv2.absdiff(original_image, cvd_image)

        # Convert the saliency map to gray image
        saliency_map_gray = cv2.cvtColor(saliency_map, cv2.COLOR_RGB2GRAY)

        # Normalize the saliency map
        saliency_map_normalized = cv2.normalize(saliency_map_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Save the saliency map
        saliency_output_path = os.path.join(output_dir, filename)
        cv2.imwrite(saliency_output_path, saliency_map_normalized)

original_image_dir = 'picture/test/'
cvd_image_dir = 'picture/simulate/'
output_dir = 'picture/saliency/'

batch_process_saliency(original_image_dir, cvd_image_dir, output_dir)