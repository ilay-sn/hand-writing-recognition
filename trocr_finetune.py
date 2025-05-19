import torch
from PIL import Image
import os # For path operations
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage # Added Features, Value, HFImage
from evaluate import load
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image as PILImage
# For mixed precision training
from torch.cuda.amp import autocast, GradScaler

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = '/home/ilayda/Downloads/trocr-small-handwritten'
#model_name = "microsoft/trocr-small-handwritten"

try:
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    print(f"Processor and model '{model_name}' loaded successfully.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    print("Ensure you have an internet connection, the model name is correct, and HF_TOKEN is set up if needed.")
    # Add dummy initializations if loading fails for notebook flow, but these won't work for training.
    if 'processor' not in locals():
        from transformers import AutoTokenizer, AutoImageProcessor
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        image_processor_hf = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        class DummyProcessor:
            def __init__(self, image_processor, tokenizer):
                self.image_processor = image_processor
                self.tokenizer = tokenizer
            def __call__(self, images, text=None, return_tensors="pt", **kwargs):
                pixel_values = self.image_processor(images, return_tensors=return_tensors, **kwargs).pixel_values
                if text is not None:
                    labels = self.tokenizer(text, return_tensors=return_tensors, padding="max_length", truncation=True, **kwargs).input_ids
                    return {"pixel_values": pixel_values, "labels": labels}
                return {"pixel_values": pixel_values}
            def batch_decode(self, *args, **kwargs): return self.tokenizer.batch_decode(*args, **kwargs)
        processor = DummyProcessor(image_processor_hf, tokenizer)
        print("Using dummy processor.")
    if 'model' not in locals():
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k", "roberta-base"
        ).to(device)
        print("Using dummy model.")

# Configure model for generation
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 32  # Max length for words might be shorter than lines
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0 # Or 1.0 for words
model.config.num_beams = 4

# image_column_name and text_column_name are defined at the end of Step 3
# image_column_name = 'image_path'
# text_column_name = 'text'

image_column_name = 'image_path'
text_column_name = 'text'

print(f"Using image column: '{image_column_name}', text column: '{text_column_name}' for preprocessing.")


def transform_examples(examples):
    # Load images from paths
    try:
        # Ensure images are in RGB format as expected by TrOCRProcessor
        images_pil = [PILImage.open(path).convert("RGB") for path in examples[image_column_name]]
    except Exception as e:
        print(f"Error loading image for paths {examples[image_column_name]}: {e}")
        # Handle error: e.g., return None or skip, or use a placeholder image
        # For simplicity, if an image fails, this will error out.
        # Robust handling would involve filtering these out earlier or providing a placeholder.
        raise

    texts = examples[text_column_name]

    inputs = processor(images=images_pil, text=texts, padding="max_length", truncation=True)
    # The processor output for text is typically 'input_ids'.
    # For Seq2Seq models, these 'input_ids' (from text) become the 'labels'.
    inputs['labels'] = inputs['labels']  # .clone() is not strictly necessary here
    return inputs

train_dataset = Dataset.from_file('data/train.hf/data-00000-of-00001.arrow')
eval_dataset = Dataset.from_file('data/eval.hf/data-00000-of-00001.arrow')
test_dataset = Dataset.from_file('data/test.hf/data-00000-of-00001.arrow')

# Apply the transformation
# This will apply the transform when data is accessed, which is memory efficient.
train_dataset.set_transform(transform_examples)
eval_dataset.set_transform(transform_examples)
test_dataset.set_transform(transform_examples)

# Verify an example from the transformed dataset
# (This will now load the image and process it)
try:
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print("\nSample from transformed dataset:")
        print({k: v.shape if hasattr(v, 'shape') else v for k, v in sample.items()})
        print(f"Decoded labels: {processor.tokenizer.decode(sample['labels'])}")
        # To display the image (optional):
        # loaded_image = PILImage.open(train_dataset.data['image_path'][0]) # Access original path
        # display(loaded_image)
    else:
        print("Train dataset is empty, cannot show sample.")
except Exception as e:
    print(f"Error accessing sample from transformed dataset: {e}")
    print("This might happen if the first image path in the dataset is invalid or processing failed.")


cer_metric = load("cer")
wer_metric = load("wer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    # pred_ids are token IDs, no need to replace -100 here unless model outputs it
    # Forcing pad_token_id for any out-of-vocab predictions from generate, though usually not needed
    # pred_ids[pred_ids < 0] = processor.tokenizer.pad_token_id
    # pred_ids[pred_ids >= processor.tokenizer.vocab_size] = processor.tokenizer.pad_token_id


    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}

batch_size = 4 # Start with 4, can increase to 8 if using T4x2 and memory allows
gradient_accumulation_steps = 1

training_args = Seq2SeqTrainingArguments(
    output_dir="saved_models",
    predict_with_generate=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    fp16=torch.cuda.is_available(),
    num_train_epochs=10,          # Requirement: at least 10
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="cer",  # Use CER (lower is better)
    greater_is_better=False,
    # report_to="tensorboard", # Optional: Kaggle supports TensorBoard
    # optim="adamw_torch_fused", # If PyTorch 2.0+
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,  # Pass the image_processor part
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

print("Starting training...")
try:
    train_results = trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    print("Model, metrics, and state saved.")

except Exception as e:
    print(f"Error during training: {e}")
    if "CUDA out of memory" in str(e):
        print(
            "CUDA out of memory. Try reducing 'per_device_train_batch_size' or increasing 'gradient_accumulation_steps'.")
    # Consider other specific errors like issues with data loading during training
    elif "PIL.UnidentifiedImageError" in str(e):
        print("Error: PIL.UnidentifiedImageError during training. Some images might be corrupted.")
        print("Ensure all image files are valid and accessible.")

print("Evaluating on the test set...")

# Ensure the model is on the correct device for evaluation
if 'trainer' in locals() and hasattr(trainer, 'model') and trainer.model is not None:
    model_to_eval = trainer.model.to(device)
    print("Using model from trainer for evaluation.")
else:
    print("Trainer or trained model not available. Loading base model or from output_dir for evaluation.")
    # Attempt to load from output_dir if training happened and saved a model
    saved_model_path = training_args.output_dir
    if os.path.exists(os.path.join(saved_model_path, "pytorch_model.bin")):
        print(f"Loading model from {saved_model_path}")
        model_to_eval = VisionEncoderDecoderModel.from_pretrained(saved_model_path).to(device)
        # Processor should be loaded from the same path if it was saved by trainer.save_model()
        # For simplicity, we assume the global 'processor' is still valid.
    else:
        print(f"No saved model found at {saved_model_path}. Loading base model '{model_name}' for evaluation.")
        model_to_eval = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

    # Re-configure if loaded base model
    model_to_eval.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model_to_eval.config.pad_token_id = processor.tokenizer.pad_token_id
    model_to_eval.config.vocab_size = model_to_eval.config.decoder.vocab_size
    model_to_eval.config.eos_token_id = processor.tokenizer.sep_token_id
    model_to_eval.config.max_length = 32
    model_to_eval.config.early_stopping = True
    model_to_eval.config.no_repeat_ngram_size = 3
    model_to_eval.config.length_penalty = 2.0  # or 1.0
    model_to_eval.config.num_beams = 4

# Option 1: Use Trainer's predict method
if 'trainer' in locals() and hasattr(trainer, 'model') and trainer.model is not None and len(test_dataset) > 0:
    print("Using trainer.predict for evaluation...")
    try:
        test_predictions = trainer.predict(test_dataset)
        print("Test set CER:", test_predictions.metrics["test_cer"])
        print("Test set WER:", test_predictions.metrics["test_wer"])
        trainer.log_metrics("test", test_predictions.metrics)
        trainer.save_metrics("test", test_predictions.metrics)

        # Target: CER <= 7% (0.07) and WER <= 15% (0.15) - Note these were for lines
        if test_predictions.metrics["test_cer"] <= 0.07:
            print("Target CER met for words!")
        else:
            print("Target CER NOT met for words.")

        if test_predictions.metrics["test_wer"] <= 0.15:
            print("Target WER met for words!")
        else:
            print("Target WER NOT met for words.")

    except Exception as e:
        print(f"Error during trainer.predict: {e}")
        print("Consider manual evaluation loop.")

elif len(test_dataset) == 0:
    print("Test dataset is empty. Cannot evaluate.")
else:
    # Option 2: Manual evaluation loop (if trainer is not available or for more control)
    print(
        "Trainer not available or test_dataset empty for trainer.predict, performing manual evaluation loop if possible...")
    all_preds_manual = []
    all_labels_manual = []

    model_to_eval.eval()  # Set model to evaluation mode

    from torch.utils.data import DataLoader

    # The test_dataset already has the transform set.
    # default_data_collator will be used by DataLoader if we pass it from trainer,
    # or we can use a simple custom one if needed.
    # For Seq2Seq, the labels are already padded by the processor.

    # We need a collate function that can handle the output of our transformed dataset
    # The `default_data_collator` should work as it handles dicts of tensors.
    test_dataloader_manual = DataLoader(
        test_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=default_data_collator
    )

    with torch.no_grad():
        for batch in test_dataloader_manual:
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            generated_ids = model_to_eval.generate(pixel_values)

            pred_strs_manual = processor.batch_decode(generated_ids, skip_special_tokens=True)
            labels[labels == -100] = processor.tokenizer.pad_token_id
            label_strs_manual = processor.batch_decode(labels, skip_special_tokens=True)

            all_preds_manual.extend(pred_strs_manual)
            all_labels_manual.extend(label_strs_manual)

    if all_preds_manual and all_labels_manual:
        final_cer_manual = cer_metric.compute(predictions=all_preds_manual, references=all_labels_manual)
        final_wer_manual = wer_metric.compute(predictions=all_preds_manual, references=all_labels_manual)
        print(f"Manual Test Set CER: {final_cer_manual}")
        print(f"Manual Test Set WER: {final_wer_manual}")
    else:
        print("No predictions made in manual loop, check data loading and preprocessing for test set.")

print(f"Fine-tuned model and processor are saved in: {training_args.output_dir}")
# This directory (/kaggle/working/trocr-iam-word-finetuned) can be saved as Kaggle output.
# Click "Save Version" on your Kaggle notebook (top right), choose "Save & Run All (Commit)".
# After it runs, the contents of /kaggle/working/ will be in the "Output" tab of the version.