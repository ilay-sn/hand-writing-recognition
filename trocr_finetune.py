import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import VisionEncoderDecoderModel
from data_process import IAMDataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils import is_valid_image, compute_metrics
from transformers import TrOCRProcessor
from transformers import default_data_collator

df = pd.read_fwf('/home/ilayda/Downloads/iam_handwriting_word_database/iam_words/words.txt', header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)

# some file names end with jp instead of jpg, let's fix this
df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)

image_root = '/home/ilayda/Downloads/iam_handwriting_word_database/iam_words/words/'

train_df, test_df = train_test_split(df, test_size=0.2)
# we reset the indices to start from zero

train_df = train_df[train_df.apply(lambda row: is_valid_image(row, image_root), axis=1)].reset_index(drop=True)
test_df = test_df[test_df.apply(lambda row: is_valid_image(row, image_root), axis=1)].reset_index(drop=True)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

model_name = '/home/ilayda/Downloads/trocr-small-handwritten'
model = VisionEncoderDecoderModel.from_pretrained(model_name).to("cuda")
processor = TrOCRProcessor.from_pretrained(model_name)

train_dataset = IAMDataset(root_dir='/home/ilayda/Downloads/iam_handwriting_word_database/iam_words/words/',
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir='/home/ilayda/Downloads/iam_handwriting_word_database/iam_words/words/',
                           df=test_df,
                           processor=processor)


# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4


training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True,
    output_dir="saved_models",
    logging_steps=20,
    save_strategy="epoch",
    eval_steps=200,
    dataloader_pin_memory=False,
    num_train_epochs=15,
    eval_strategy="epoch"
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
trainer.train()
