from keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer

text = "The quick brown fox jumps over the lazy dog."

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
tokens = tokenizer.texts_to_sequences([text])[0]
print("The tokenized text:", tokens)


stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in tokens]
print("The stemmed tokens:", stemmed_tokens)
