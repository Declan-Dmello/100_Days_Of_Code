import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split


def simple_lstm_sentiment():
    # Sample dataset
    texts = [
        "i love this movie",
        "great awesome film highly recommend",
        "excellent acting",
        "terrible waste of time",
        "poor quality disappointing",
        "not worth watching",
        "amazing movie love it",
        "worst film ever",
        "fantastic great watch",
        "hate it terrible",
        "brilliant performance",
        "boring and bad",
        "really enjoyed it",
        "waste of money",
        "outstanding film",
        "This movie is terrible",
        "The character were great"

    ]

    # Labels: 1 for positive, 0 for negative
    labels = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,0,1])


    max_words = 1000
    max_len = 20


    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    X = pad_sequences(sequences, maxlen=max_len)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    model = Sequential([
        Embedding(max_words, 32, input_length=max_len),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("Training the model...")
    model.fit(X_train, y_train,
              epochs=10,
              batch_size=4,
              validation_split=0.2,
              verbose=1)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    def predict_sentiment(text):
        new_sequence = tokenizer.texts_to_sequences([text])
        new_padded = pad_sequences(new_sequence, maxlen=max_len)

        prediction = model.predict(new_padded)[0][0]
        return "Positive" if prediction > 0.5 else "Negative", prediction

    print("\nTesting with new reviews:")
    test_reviews = [
        "this movie is awesome",
        "complete waste of time",
        "really enjoyed watching",
        "terrible movie dont watch"
    ]

    for review in test_reviews:
        sentiment, confidence = predict_sentiment(review)
        print(f"\nReview: '{review}'")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    simple_lstm_sentiment()