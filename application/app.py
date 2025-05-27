from flask import Flask, render_template, request
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os
import easyocr
import torchvision.transforms as transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and processor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained('../saved_models/checkpoint-8908')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# OCR Reader
reader = easyocr.Reader(['en'], gpu=True)  # GPU-enabled

# Preprocess
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # or keep aspect ratio if needed
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    text_output = ""
    image_path = ""
    recognized_words = []

    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            image = Image.open(image_path).convert("RGB")

            bounds = reader.readtext(image_path, detail=1)  # (bbox, text, conf)

            for bbox, _, _ in bounds:
                # Get cropped word image
                x_min = int(min([pt[0] for pt in bbox]))
                y_min = int(min([pt[1] for pt in bbox]))
                x_max = int(max([pt[0] for pt in bbox]))
                y_max = int(max([pt[1] for pt in bbox]))

                word_img = image.crop((x_min, y_min, x_max, y_max))

                # Preprocess and predict
                pixel_values = processor(images=word_img, return_tensors="pt").pixel_values.to("cuda")

                with torch.no_grad():
                    generated_ids = model.generate(pixel_values)
                pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                recognized_words.append(pred_text)

    return render_template('index.html', extracted_text=" ".join(recognized_words), image_path=image_path)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
