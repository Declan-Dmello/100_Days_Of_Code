import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
import os
import wget

if not os.path.exists('glove.6B.100d.txt'):
    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    wget.download(url)
    print("Please unzip the downloaded file and ensure glove.6B.100d.txt is in the current directory")

glove_file = 'glove.6B.100d.txt'
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

glove_model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

king_vector = glove_model['king']
print("Vector for 'king':")

similar_words = glove_model.most_similar('woman')
print("\nMost similar words to 'woman':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.3f}")


result = glove_model.most_similar(positive=['woman', 'king'], negative=['man'])
print("\nAnalogy: woman is to king as man is to ?")
print(result)