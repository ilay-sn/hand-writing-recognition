# train_trocr_finetune.py

import os
import yaml
import torch
from datasets import load_from_disk
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AdamW,
    get_scheduler
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image as PILImage

def main():
    # 1. Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load HF datasets from disk
    data_cfg = cfg["data"]
    train_ds = load_from_disk(data_cfg["train_dataset_dir"])
    val_ds   = load_from_disk(data_cfg["eval_dataset_dir"])
    test_ds  = load_from_disk(data_cfg["test_dataset_dir"])

    # Optional subsampling for quick debugging
    def maybe_sub(ds, pct):
        return ds.select(range(int(len(ds) * pct / 100))) if pct < 100 else ds

    train_ds = maybe_sub(train_ds, data_cfg["train_subset_pct"])
    val_ds   = maybe_sub(val_ds,   data_cfg["val_subset_pct"])
    test_ds  = maybe_sub(test_ds,  data_cfg["test_subset_pct"])

    # 4. Initialize processor & model
    proc_cfg  = cfg["processor"]
    model_cfg = cfg["model"]
    processor = TrOCRProcessor.from_pretrained(
        model_cfg["name_or_path"],
        use_fast=proc_cfg["use_fast"]
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        model_cfg["name_or_path"]
    ).to(device)

    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.eos_token_id

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten", use_fast=False)
    processor.image_processor.image_channel_first = False

    def preprocess(batch):
        images = [PILImage.open(p).convert("RGB") for p in batch[data_cfg["image_column"]]]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values

        # Pad *all* sequences to max_length:
        labels = processor.tokenizer(
            batch[data_cfg["text_column"]],
            padding="max_length",
            max_length=cfg["preprocessing"]["max_length"],
            truncation=cfg["preprocessing"]["truncation"],
            return_tensors="pt"
        ).input_ids

        batch["pixel_values"] = pixel_values.tolist()
        batch["labels"]       = labels.tolist()
        return batch

    # Apply preprocessing and drop original columns
    remove_cols = [data_cfg["image_column"], data_cfg["text_column"]]
    train_ds = train_ds.map(
        preprocess,
        batched=True,
        batch_size=cfg["training"]["per_device_train_batch_size"],
        remove_columns=remove_cols
    )
    val_ds = val_ds.map(
        preprocess,
        batched=True,
        batch_size=cfg["training"]["per_device_eval_batch_size"],
        remove_columns=remove_cols
    )
    test_ds = test_ds.map(
        preprocess,
        batched=True,
        batch_size=cfg["training"]["per_device_eval_batch_size"],
        remove_columns=remove_cols
    )
    train_ds.set_format(type="torch", columns=["pixel_values", "labels"])
    val_ds.  set_format(type="torch", columns=["pixel_values", "labels"])
    test_ds. set_format(type="torch", columns=["pixel_values", "labels"])
    # 6. DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["per_device_train_batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["per_device_eval_batch_size"],
        shuffle=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["per_device_eval_batch_size"],
        shuffle=False
    )

    # 7. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr= 0.0001, #cfg["optimizer"]["params"]["lr"],
        weight_decay=0.0001, #cfg["optimizer"]["params"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["training"]["num_train_epochs"]
    lr_scheduler = get_scheduler(
        cfg["training"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=cfg["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    # 8. Training & validation loop
    for epoch in range(cfg["training"]["num_train_epochs"]):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['num_train_epochs']}")
        for batch in loop:
            pixel_values = torch.tensor(batch["pixel_values"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0.0
        for batch in val_loader:
            pixel_values = torch.tensor(batch["pixel_values"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device)
            with torch.no_grad():
                val_loss += model(pixel_values=pixel_values, labels=labels).loss.item()
        avg_val = val_loss / len(val_loader)
        print(f"→ Avg validation loss: {avg_val:.4f}")

        # Save epoch checkpoint
        ckpt_dir = os.path.join(model_cfg["output_dir"], f"checkpoint-epoch{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)

    # 9. Final test evaluation
    model.eval()
    test_loss = 0.0
    for batch in test_loader:
        pixel_values = torch.tensor(batch["pixel_values"]).to(device)
        labels = torch.tensor(batch["labels"]).to(device)
        with torch.no_grad():
            test_loss += model(pixel_values=pixel_values, labels=labels).loss.item()
    print(f"Test loss: {test_loss / len(test_loader):.4f}")

    # 10. Save final model & processor
    os.makedirs(model_cfg["output_dir"], exist_ok=True)
    model.save_pretrained(model_cfg["output_dir"])
    processor.save_pretrained(model_cfg["output_dir"])
    print("▶ Training complete. Model saved to", model_cfg["output_dir"])

if __name__ == "__main__":
    main()
