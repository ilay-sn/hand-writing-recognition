from datasets import Dataset, DatasetDict, Features, Value
from PIL import Image as PILImage  # Use a distinct alias for PIL.Image
import os
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor

"""
Dataset can be downloaded from: "nibinv23/iam-handwriting-word-database" 
"""

# Define base paths for the Kaggle dataset
BASE_DATASET_PATH = '/home/ilayda/Downloads/iam_handwriting_word_database/iam_words/'
WORDS_TXT_PATH = os.path.join(BASE_DATASET_PATH, "words.txt")
IMAGE_BASE_PATH = os.path.join(BASE_DATASET_PATH, "words")  # Path to the directory containing a01, a02, etc.

print(f"Looking for words.txt at: {WORDS_TXT_PATH}")
print(f"Base image directory: {IMAGE_BASE_PATH}")

all_image_paths = []
all_texts = []

if not os.path.exists(WORDS_TXT_PATH):
    print(f"ERROR: words.txt not found at {WORDS_TXT_PATH}")
    # Define empty datasets to prevent immediate crashes in later cells
    train_dataset = Dataset.from_dict({"image_path": [], "text": []})
    eval_dataset = Dataset.from_dict({"image_path": [], "text": []})
    test_dataset = Dataset.from_dict({"image_path": [], "text": []})

else:
    print("Parsing words.txt...")
    with open(WORDS_TXT_PATH, "r") as f:
        for line in f:
            if line.startswith("#"):  # Skip comment lines
                continue

            parts = line.strip().split()
            if len(parts) < 9:  # Ensure the line has enough parts
                continue

            word_id = parts[0]
            segmentation_status = parts[1]
            text_label = parts[-1]  # The actual word is the last part

            if segmentation_status == "ok":
                # Construct the image path: e.g., words/a01/a01-000u/a01-000u-00-00.png
                # word_id might be 'a01-000u-00-00'
                id_parts = word_id.split('-')
                if len(id_parts) < 2:
                    # print(f"Warning: Skipping malformed word_id {word_id}")
                    continue

                # Path: .../iam_words/words/a01/a01-000u/a01-000u-00-00.png
                image_path = os.path.join(IMAGE_BASE_PATH, id_parts[0], f"{id_parts[0]}-{id_parts[1]}",
                                          f"{word_id}.png")

                if os.path.exists(image_path):  # Check if the image file actually exists
                    all_image_paths.append(image_path)
                    all_texts.append(text_label)
                # else:
                # print(f"Warning: Image file not found: {image_path}")
            # else:
            # print(f"Skipping word {word_id} due to segmentation status: {segmentation_status}")

    print(f"Found {len(all_image_paths)} valid word images and labels.")

    if not all_image_paths:
        print("ERROR: No valid image paths found. Check dataset structure and parsing logic.")
        # Define empty datasets
        train_dataset = Dataset.from_dict({"image_path": [], "text": []})
        eval_dataset = Dataset.from_dict({"image_path": [], "text": []})
        test_dataset = Dataset.from_dict({"image_path": [], "text": []})
    else:
        # Split data: 80% train, 10% validation, 10% test
        # First split: train+validation vs test
        train_val_paths, test_paths, train_val_texts, test_texts = train_test_split(
            all_image_paths, all_texts, test_size=0.1, random_state=42, stratify=None
            # Stratify can be difficult with text
        )
        # Second split: train vs validation
        train_paths, val_paths, train_texts, val_texts = train_test_split(
            train_val_paths, train_val_texts, test_size=0.111, random_state=42  # 0.111 of 0.9 is approx 0.1 of total
        )

        print(f"Training samples: {len(train_paths)}")
        print(f"Validation samples: {len(val_paths)}")
        print(f"Test samples: {len(test_paths)}")

        # Create Hugging Face Dataset objects
        # Note: We are storing image_path now, not the loaded image.
        # The image loading will happen in the preprocessing transform.
        train_dataset = Dataset.from_dict({"image_path": train_paths, "text": train_texts})
        eval_dataset = Dataset.from_dict({"image_path": val_paths, "text": val_texts})
        test_dataset = Dataset.from_dict({"image_path": test_paths, "text": test_texts})

        # Define features (optional but good practice, helps with type consistency)
        # features = Features({
        #     'image_path': Value(dtype='string'), # We'll load images on the fly
        #     'text': Value(dtype='string')
        # })
        # train_dataset = train_dataset.cast(features)
        # eval_dataset = eval_dataset.cast(features)
        # test_dataset = test_dataset.cast(features)

# Final check
print("Dataset splits created:")
print(f"Train: {train_dataset}")
print(f"Validation: {eval_dataset}")
print(f"Test: {test_dataset}")


# Filter out problematic data (e.g., empty text after splitting, though unlikely here)
def is_valid_sample(sample):
    return isinstance(sample['image_path'], str) and \
        os.path.exists(sample['image_path']) and \
        isinstance(sample['text'], str) and \
        len(sample['text'].strip()) > 0


train_dataset = train_dataset.filter(is_valid_sample)
eval_dataset = eval_dataset.filter(is_valid_sample)
test_dataset = test_dataset.filter(is_valid_sample)

print(f"\nAfter filtering: Number of training samples: {len(train_dataset)}")
print(f"After filtering: Number of validation samples: {len(eval_dataset)}")
print(f"After filtering: Number of test samples: {len(test_dataset)}")

train_dataset.save_to_disk("data/train.hf")
eval_dataset.save_to_disk("data/eval.hf")
test_dataset.save_to_disk("data/test.hf")

# Define column names for later use
image_column_name = 'image_path'
text_column_name = 'text'
