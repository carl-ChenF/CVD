import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 加载预训练的 BLIP 模型和处理器
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_image_caption(image_path, prompt=''):
    # 加载并处理图像
    image = Image.open(image_path)  # 替换为你的图像路径

    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # 生成图像描述
    with torch.no_grad():
        output = model.generate(**inputs)

    # 解码生成的描述
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

image_path = 'picture/'
print("生成的描述:", caption)
