from PIL import Image
import os
import evaluate
from transformers import TrOCRProcessor

model_name = '/home/ilayda/Downloads/trocr-small-handwritten'
processor = TrOCRProcessor.from_pretrained(model_name)


def is_valid_image(row, root_dir):
    try:
        file_name = row['file_name'].split()[0]
        subfolder1 = file_name[:3]
        subfolder2 = file_name.rsplit('-', 2)[0]
        file_path = os.path.join(root_dir, subfolder1, subfolder2, file_name + '.png')
        with Image.open(file_path) as img:
            return img.size != (1, 1)
    except:
        return False


def compute_metrics(pred):
    cer_metric = evaluate.load("cer")
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}