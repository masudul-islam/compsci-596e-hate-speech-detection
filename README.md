# Hate Speech Detection

## Project Overview
This project develops a machine learning model to detect hate speech in tweets. It utilizes a pre-trained BERT model fine-tuned on a dataset containing tweets labeled as hate speech, offensive language, or neither.

## Installation

### Prerequisites
- Python 3.8+ or higher 
- pip (Python package installer)

### Setup
To set up the project, follow these steps to clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/masudul-islam/compsci-596e-hate-speech-detection.git
cd compsci-596e-hate-speech-detection
pip install -r requirements.txt
```

## Code Formatting
In this project, I use black and isort for code formatting and import sorting to maintain a clean and consistent codebase.

To run black, use the following command:
```bash
black .
```
To run isort, use the following command:
```bash
isort .
```

## CLI Tool
You can test the CLI tool with the following example commands:

```bash
python predict_cli.py "I love this new game, it's fantastic."
python predict_cli.py "I gotta go now, bitch."
python predict_cli.py "Dumb bitches do dumb things."
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
