import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def simple_text_classifier():
    data = {
        'text': [
            "i love this product",
            "great purchase highly recommend",
            "excellent service",
            "terrible waste of money",
            "poor quality disappointing",
            "not worth it",
            "amazing product love it",
            "worst purchase ever",
            "fantastic great buy",
            "hate it terrible",
            "The Market is horrible today",
            "The Market is nice today",
            "The sky is mesmerizing"
        ],
        'sentiment': [1, 1, 1, 0, 0, 0, 1, 0, 1, 0,0,1,1]
    }
    df = pd.DataFrame(data)
    print(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['sentiment'],
        test_size=0.2,
        random_state=42
    )

    vectorizer = CountVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    print("Numerical Vectors")
    print(X_train_vectors)

    classifier = MultinomialNB()
    classifier.fit(X_train_vectors, y_train)

    def predict_sentiment(text):
        text_vector = vectorizer.transform([text])
        prediction = classifier.predict(text_vector)[0]
        return "Positive" if prediction == 1 else "Negative"

    print("Test with some new statements:")
    test_reviews = [
        "this is really great",
        "total disappointment",
        "The Market is great",
        "The sky is beautiful"
    ]

    for review in test_reviews:
        sentiment = predict_sentiment(review)
        print(f"Review: '{review}' -> {sentiment}")


if __name__ == "__main__":
    simple_text_classifier()