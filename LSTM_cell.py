import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer

text = "the quick brown fox jumps over the lazy dog"


chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)


sequence_length = 10
step = 1

sequences = []
next_chars = []

for i in range(0, len(text) - sequence_length, step):
    sequences.append(text[i:i + sequence_length])
    next_chars.append(text[i + sequence_length])


X = np.zeros((len(sequences), sequence_length, vocab_size), dtype=np.bool_)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool_)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1




model = Sequential()
model.add(LSTM(128, input_shape=(sequence_length, vocab_size)))
model.add(Dropout(0.2))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=64, epochs=50, verbose=1)




def generate_text(seed_text, num_generate=50):
    generated = seed_text
    for _ in range(num_generate):

        x = np.zeros((1, sequence_length, vocab_size))
        for t, char in enumerate(generated[-sequence_length:]):
            x[0, t, char_to_idx[char]] = 1

        preds = model.predict(x, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = idx_to_char[next_index]

        generated += next_char
    return generated

# Example usage
seed_text = "the quick"
generated_text = generate_text(seed_text)
print(f"Seed text: {seed_text}")
print(f"Generated text: {generated_text}")