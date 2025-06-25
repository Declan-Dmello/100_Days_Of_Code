import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
#from keras import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np


class SimpleNLPTransfer:
    def __init__(self, max_words=10000, max_len=100, embedding_dim=100):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None
        self.word_index = None

    def load_glove_embeddings(self, glove_path):

        embeddings_index = {}
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print(f'Found {len(embeddings_index)} word vectors.')
        return embeddings_index

    def prepare_embedding_matrix(self, embeddings_index):

        embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
        for word, i in self.word_index.items():
            if i < self.max_words:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def prepare_data(self, texts):

        # Fit tokenizer on texts
        self.tokenizer.fit_on_texts(texts)
        self.word_index = self.tokenizer.word_index

        sequences = self.tokenizer.texts_to_sequences(texts)

        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        return padded_sequences

    def build_model(self, embedding_matrix):

        model = Sequential([
            Embedding(self.max_words, self.embedding_dim,
                      weights=[embedding_matrix],
                      input_length=self.max_len,
                      trainable=False),

            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.model = model
        return model

    def train(self, X_train, y_train, validation_data=None, epochs=10, batch_size=32):
        """
        Train the model

        Args:
            X_train: Training sequences
            y_train: Training labels
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        return history

    def predict(self, texts):
        """
        Make predictions on new texts

        Args:
            texts: List of text samples to predict
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        return self.model.predict(padded_sequences)







# Example usage
def example_usage():
    # Sample data
    texts = [
        "This movie is amazing and I loved it",
        "This was a terrible waste of time",
        "Really enjoyed watching this film",
        "I would never recommend this movie",
        "One of the best movies I've seen"
    ]
    labels = np.array([1, 0, 1, 0, 1])  # 1 for positive, 0 for negative

    # Initialize model
    clf = SimpleNLPTransfer()

    # Prepare data
    X = clf.prepare_data(texts)

    # Load GloVe embeddings (you need to download these separately)
    # embeddings_index = clf.load_glove_embeddings('path_to_glove.txt')
    # embedding_matrix = clf.prepare_embedding_matrix(embeddings_index)

    # For demonstration, create a random embedding matrix
    embedding_matrix = np.random.randn(clf.max_words, clf.embedding_dim)

    # Build and train model
    clf.build_model(embedding_matrix)
    history = clf.train(X, labels, epochs=5, batch_size=2)

    # Make predictions
    new_texts = ["This was a fantastic movie!"]
    predictions = clf.predict(new_texts)
    return predictions

preds = example_usage()
print(preds)

if __name__ == "__main__":
    example_usage()