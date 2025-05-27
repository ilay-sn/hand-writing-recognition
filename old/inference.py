from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import requests

# load image from the IAM database
url = '/home/ilayda/Downloads/Screenshot_20250525_122321_Samsung Notes.jpg'
image = Image.open(url).convert("RGB")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained('saved_models/checkpoint-8908')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)