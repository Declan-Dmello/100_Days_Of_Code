import gensim
from gensim.models import Word2Vec
from nltk.corpus import brown
#import nltk

#nltk.download('brown')
sentences = brown.sents()

model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, workers=4)

king_vector = model.wv['king']
print("Vector for 'king':", king_vector)


similar_words = model.wv.most_similar('woman')
print("\nMost similar words to 'woman':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.3f}")


result = model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
print("\nAnalogy: woman is to king as man is to ?")
print(result)

