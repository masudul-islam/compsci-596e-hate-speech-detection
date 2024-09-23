# Hate Speech Detection System

## Project Overview
This project develops a machine learning model to detect hate speech in tweets. It utilizes a pre-trained BERT model fine-tuned on a dataset containing tweets labeled as hate speech, offensive language, or neither.

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
Clone the repository and install the required packages:

```bash
   git clone https://github.com/masudul-islam/compsci-596e-hate-speech-detection.git
   cd hatespeech-detection
   pip install -r requirements.txt
```

## CLI Tool
To predict the classification of a tweet from the command line:

```bash
   python predict_cli.py "Your tweet here"
```

## Flask Application
To run the Flask server:

```bash
   python app.py
   ```
Send a POST request to predict hate speech in a tweet:
```bash 
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"tweet":"Your tweet text here"}'
```
