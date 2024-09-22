# Import libraries
import os
import time
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load the model from the checkpoint
model_path = './results/checkpoint-4957'
model = BertForSequenceClassification.from_pretrained(model_path)

# Load the tokenizer from the original pretrained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)

def predict_hate_speech(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    labels = ['hate_speech', 'offensive', 'neutral']
    return labels[prediction.item()]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    tweet = data['tweet']
    prediction = predict_hate_speech(tweet)
    return jsonify({'tweet': tweet, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)