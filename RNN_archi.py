import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


text = """
The Night We Met - Lord Huron 
"""

chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

sequence_length = 10
sequences = []
next_chars = []

for i in range(len(text) - sequence_length):
    sequences.append(text[i:i + sequence_length])
    next_chars.append(text[i + sequence_length])

X = np.zeros((len(sequences), sequence_length, vocab_size))
y = np.zeros((len(sequences), vocab_size))

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

model = Sequential([
    SimpleRNN(128, input_shape=(sequence_length, vocab_size)),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(X, y,
                    batch_size=32,
                    epochs=100,
                    validation_split=0.1)


# Function to generate text
def generate_text(seed_text, next_chars=50):
    generated = seed_text

    for _ in range(next_chars):
        x = np.zeros((1, sequence_length, vocab_size))
        for t, char in enumerate(generated[-sequence_length:]):
            x[0, t, char_to_idx[char]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = np.argmax(predictions)
        next_char = idx_to_char[next_index]

        generated += next_char

    return generated


seed = text[:sequence_length]
generated_text = generate_text(seed)
print("\nSeed text:", seed)
print("Generated text:", generated_text)


def generate_text_with_temperature(seed_text, next_chars=50, temperature=0.5):
    generated = seed_text

    for _ in range(next_chars):
        x = np.zeros((1, sequence_length, vocab_size))
        for t, char in enumerate(generated[-sequence_length:]):
            x[0, t, char_to_idx[char]] = 1

        predictions = model.predict(x, verbose=0)[0]

        predictions = np.log(predictions) / temperature
        exp_predictions = np.exp(predictions)
        predictions = exp_predictions / np.sum(exp_predictions)

        next_index = np.random.choice(range(len(predictions)), p=predictions)
        next_char = idx_to_char[next_index]

        generated += next_char

    return generated


print("\nGenerated text with temp 0.2:",
      generate_text_with_temperature(seed, temperature=0.2))
print("\nGenerated text with temp 1.0:",
      generate_text_with_temperature(seed, temperature=1.0))
print("\nGenerated text with temp 2.0:",
      generate_text_with_temperature(seed, temperature=2.0))