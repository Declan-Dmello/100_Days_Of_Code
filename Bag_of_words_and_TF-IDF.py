from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "The cat sits on the mat",
    "The dog runs in the park",
    "Cats and dogs are pets"
]


bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(documents)

print("Vocab:", bow_vectorizer.vocabulary_)
print("\nFeatures:", bow_vectorizer.get_feature_names_out())
print("\nBag of words Matrix:\n", bow_matrix.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\n\n\nVocab:", tfidf_vectorizer.vocabulary_)
print("\nFeature :", tfidf_vectorizer.get_feature_names_out())
print("\nTFIDF Matrix:\n", tfidf_matrix.toarray())






word = "cat"
if word in bow_vectorizer.vocabulary_:
    word_idx = bow_vectorizer.vocabulary_[word]
    print(f"\nFrequency of '{word}' in each document (BOW):", bow_matrix.toarray()[:, word_idx])
    print(f"Importance of '{word}' in each document (TF-IDF):", tfidf_matrix.toarray()[:, word_idx])