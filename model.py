# Import the libraries
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

# Load the dataset
url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataframe
print(df.head())

# Preprocess the dataset: map classes and filter
df = df[["tweet", "class"]]
label_dict = {0: "hate_speech", 1: "offensive", 2: "neutral"}
df["class"] = df["class"].map(label_dict)

# Rename 'class' to 'labels' for clarity and model compatibility
df.rename(columns={"class": "labels"}, inplace=True)
label_map = {"hate_speech": 0, "offensive": 1, "neutral": 2}
df["labels"] = df["labels"].map(label_map)

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert the data into a HuggingFace Dataset format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load the pre-trained BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["tweet"], truncation=True, padding="max_length", max_length=128
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set the format of the dataset to PyTorch tensors
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # Lower number of epoch to reduce training time
    per_device_train_batch_size=4,  # Smaller number of batch size to reduce training time
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation results:", results)
