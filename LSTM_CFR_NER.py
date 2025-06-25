import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_crf import CRF


def create_simple_lstm_crf(max_len, vocab_size, num_tags):
    input_layer = Input(shape=(max_len,))
    embedding = Embedding(vocab_size, 100, mask_zero=True)(input_layer)
    bilstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    crf = CRF(num_tags)
    output = crf(bilstm)
    model = Model(input_layer, output)
    model.compile(optimizer='adam',
                  loss=crf.loss,
                  metrics=[crf.accuracy])
    return model


def prepare_data(sentences, labels, word2idx, tag2idx, max_len):
    X = [[word2idx.get(w, word2idx['UNK']) for w in s] for s in sentences]
    y = [[tag2idx[t] for t in l] for l in labels]

    X = pad_sequences(X, maxlen=max_len, padding='post')
    y = pad_sequences(y, maxlen=max_len, padding='post')

    return X, y


# Example of how to use the model
if __name__ == "__main__":
    # Sample data
    sentences = [
        ['John', 'lives', 'in', 'New', 'York'],
        ['Apple', 'releases', 'new', 'iPhone']
    ]

    labels = [
        ['B-PER', 'O', 'O', 'B-LOC', 'I-LOC'],
        ['B-ORG', 'O', 'O', 'B-PROD']
    ]

    # Sample dictionaries
    word2idx = {'UNK': 0, 'John': 1, 'lives': 2, 'in': 3, 'New': 4, 'York': 5,
                'Apple': 6, 'releases': 7, 'new': 8, 'iPhone': 9}

    tag2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4,
               'B-ORG': 5, 'I-ORG': 6, 'B-PROD': 7, 'I-PROD': 8}

    # Model parameters
    MAX_LEN = 10
    VOCAB_SIZE = len(word2idx)
    NUM_TAGS = len(tag2idx)

    # Prepare data
    X, y = prepare_data(sentences, labels, word2idx, tag2idx, MAX_LEN)

    # Create and train model
    model = create_simple_lstm_crf(MAX_LEN, VOCAB_SIZE, NUM_TAGS)
    model.fit(X, y, epochs=5, batch_size=2)