# Import libraries
import subprocess

# List of sample tweets to classify
sample_tweets = [
    "I love this new game, it's fantastic!",
    "I gotta go now, bitch.",
    "Dumb bitches do dumb things.",
]

# Iterate through the sample tweets and call the CLI tool
for tweet in sample_tweets:
    print(f"Testing tweet: {tweet}")
    # Call the CLI tool and capture the output
    result = subprocess.run(['/Library/Frameworks/Python.framework/Versions/3.12/bin/python3', 'predict_cli.py', tweet], capture_output=True, text=True)

    print("Output:", result.stdout)
