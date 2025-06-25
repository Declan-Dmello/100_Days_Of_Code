import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense
from keras.preprocessing.sequence import pad_sequences

# Sample Data Preparation
# Let's assume we have sequences of integers
# For simplicity, let's create random data
num_samples = 1000
max_length = 10  # Maximum length of input sequences
num_classes = 2   # Number of output classes

# Generate random sequences and labels
X = np.random.randint(1, 100, size=(num_samples, max_length))
y = np.random.randint(0, num_classes, size=(num_samples,))







X = pad_sequences(X, maxlen=max_length)

model = Sequential()
model.add(Bidirectional(LSTM(64), input_shape=(max_length, 1)))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X = X.reshape((num_samples, max_length, 1))


model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2)


loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss}, Accuracy: {accuracy}')
