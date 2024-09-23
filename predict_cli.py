# Import libraries
import argparse
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def get_prediction(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('./results/checkpoint-4957')
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    labels = ['hate_speech', 'offensive', 'neutral']
    return labels[prediction.item()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hate Speech Detection CLI")
    parser.add_argument('tweet', type=str, help='Input tweet text for hate speech detection')
    args = parser.parse_args()

    prediction = get_prediction(args.tweet)
    print(f"Prediction: {prediction}")
