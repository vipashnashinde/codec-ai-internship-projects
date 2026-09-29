import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample Dataset
data = {
    'tweet': [
        'I love this product, absolutely amazing!',
        'Worst experience ever, totally disappointed.',
        'It is okay, nothing extraordinary.',
        'Highly recommended, fantastic quality!',
        'Terrible service, will not buy again.'
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Training a Simple Naive Bayes NLP Pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df['tweet'], df['sentiment'])

# Test Predictions
test_tweets = [
    "Truly wonderful experience!",
    "Awful service and poor quality."
]

predictions = model.predict(test_tweets)

for tweet, sentiment in zip(test_tweets, predictions):
    print(f"Tweet: '{tweet}' --> Predicted Sentiment: {sentiment}")
