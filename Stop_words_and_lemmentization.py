import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#from keras.preprocessing.text import Tokenizer


#nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])


text = "The cats are running quickly through the gardens and jumping over fences"


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

tokens = word_tokenize(text.lower())

filtered_tokens = [token for token in tokens if token not in stop_words]

lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]


print("The old text:", text)
print("After Stop word removal:", filtered_tokens)
print("After Lemmatization:", lemmatized_tokens)
