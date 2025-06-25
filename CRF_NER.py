import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn_crfsuite import CRF
import spacy


class CRFNamedEntityRecognizer:
    def __init__(self):
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.nlp = spacy.load("en_core_web_sm")

    def word2features(self, sent, i):
        word = sent[i]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit()
        }

        if i > 0:
            word1 = sent[i - 1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper()
            })
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper()
            })
        else:
            features['EOS'] = True

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [label for label in sent]

    def preprocess_text(self, text):
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        return tokens

    def train(self, texts, labels):
        X = [self.sent2features(text) for text in texts]
        y = [self.sent2labels(label) for label in labels]
        self.crf.fit(X, y)

    def predict(self, text):
        tokens = self.preprocess_text(text)
        features = self.sent2features(tokens)
        labels = self.crf.predict([features])[0]
        result = list(zip(tokens, labels))
        return result


if __name__ == "__main__":
    train_texts = [
        ["John", "works", "at", "Google", "in", "New", "York"],
        ["Microsoft", "announced", "new", "AI", "features"],
        ["Hemant","Joined","Amazon","last","Year"]
    ]
    train_labels = [
        ["B-PER", "O", "O", "B-ORG", "O", "B-LOC", "I-LOC"],
        ["B-ORG", "O", "O", "B-MISC", "O"],
        ["B-PER","O","B-ORG","O","O"]
    ]

    ner = CRFNamedEntityRecognizer()
    ner.train(train_texts, train_labels)

    test_text = "Hemant joined Salesforce last week"
    predictions = ner.predict(test_text)
    print(predictions)