"""
# Data Preprocessing
Once we have extracted our JSON files with our webcrawler, we now want to preprocess the data to 1) extract the ticker label, if available, and 2) generate a "sentimental" score from the cleaned text.
"""

# Import dependencies
import os
import json
import numpy as np
import pandas as pd
import yfinance
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



def extract_json(filename: str) -> dict:
    """Extracts JSON file and returns a dictionary"""
    with open(filename) as json_file:
        json_dict = json.load(json_file)
    
    return json_dict


def get_ticker(raw : dict):
    """Returns a ticker if it exists, else returns none"""
    # Check title for ticker first
    title = raw['submission_title'].split(' ')
    for word in title:
        if (is_valid_ticker(word)):
            # If something is a valid ticker, return the ticker
            return word

    # Check body for ticker
    body = raw['body'].split(' ')
    for word in body:
        if (is_valid_ticker(word)):
            # If something is a valid ticker, return the ticker
            return word
    
    # Return None otherwise
    return None

def is_valid_ticker(word):
    """Returns boolean if ticker is valid or not"""
    ticker = yfinance.Ticker(word)
    
    # If there is no regularMarketPrice, there is definitely not a ticker
    if ticker.info['regularMarketPrice'] == None:
        return False
    else:
        return True

def scrape_data():
    # Makes sure ./data exists in our folder tree
    if not os.path.exists("../data"):
        print("Making directory ../data...")
        os.makedirs("../data")
    else:
        print("../data already exists.")


    test_data = extract_json("../data/test/2018-09-01.json")
    print(test_data[0])
    print(test_data[np.random.randint(len(test_data))]['body'])

    """
    Before we can clean up our text, we need to define a function to grab the ticker of a post. If the ticker does not exist, the post is probably not relevant since we have no way to correlate a post with a specific stock. After confirming a valid ticker in a post, we have to define a function that will clean up the body of our text.
    """

    rand_val = np.random.randint(len(test_data))
    text = test_data[rand_val]


    rand_index = np.random.randint(len(test_data))
    text = test_data[rand_index]
    text_title = text['submission_title']
    text_body = text['body']
    analyzer = SentimentIntensityAnalyzer()
    score_title = analyzer.polarity_scores(text_title)
    score_body = analyzer.polarity_scores(text_body)
    print(f"{text_title} {str(score_title)}")
    print(f"{text_body} {str(score_body)}")


