import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx].split()[0]  # e.g., 'a03-017-04-02'
        subfolder1 = file_name[:3]  # 'a03'
        subfolder2 = file_name.rsplit('-', 2)[0]  # 'a03-017'

        file_path = os.path.join(self.root_dir, subfolder1, subfolder2, file_name + '.png')
        image = Image.open(file_path).convert("RGB")
        #print("Image size is:", image.size)
        text = self.df['file_name'][idx].split()[-1]
        # prepare image (i.e. resize + normalize)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze().to("cuda"), "labels": torch.tensor(labels).to("cuda")}
        return encoding