import torch
from transformers import BertTokenizer, BertModel
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np


def load_word2vec():
    print("Loading Word2Vec embeddings...")
    return api.load('word2vec-google-news-300')

def load_glove():
    print("Loading GloVe embeddings...")
    return api.load('glove-wiki-gigaword-300')


def find_similar_words(word, model_name, model):
    try:
        similar_words = model.most_similar(word)
        print(f"\nMost similar words to '{word}' using {model_name}:")
        for w, score in similar_words[:5]:
            print(f"{w}: {score:.4f}")
    except KeyError:
        print(f"Word not found in {model_name} vocabulary")


def analyze_analogies(word2vec, glove):
    analogies = [
        (['woman', 'king'], ['man']),
        (['paris', 'germany'], ['france']),
        (['better', 'bad'], ['good'])
    ]

    for positive, negative in analogies:
        print(f"\nAnalogy: {positive[0]} - {negative[0]} + {positive[1]}")

        print("Word2Vec:")
        try:
            result = word2vec.most_similar(positive=positive, negative=negative)
            print(f" {result[0][0]} (score: {result[0][1]:.4f})")
        except KeyError:
            print("Word doesnt exist in Word2Vec vocab")

        print("GloVe result:")
        try:
            result = glove.most_similar(positive=positive, negative=negative)
            print(f"{result[0][0]} (score: {result[0][1]:.4f})")
        except KeyError:
            print("Word doesnt exist in GloVe vocab")



word2vec = load_word2vec()
glove = load_glove()

test_words = ["Paradox", "fable", "learning", "Assumption"]

for word in test_words:
    find_similar_words(word, "Word2Vec", word2vec)
    find_similar_words(word, "GloVe", glove)

analyze_analogies(word2vec, glove)