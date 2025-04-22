import json
import os
from scipy.sparse.linalg import svds
import numpy as np

# from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

data = []
for k in range(10):
    with open(os.path.join(current_directory, "data_" + str(k) + ".json"), "r") as file:
        new_data = json.load(file)
        data = data + new_data

vectorizer = TfidfVectorizer(
    use_idf=True, strip_accents="unicode", lowercase=True, stop_words="english"
)
doc_terms = []
for d in data:
    terms = d["description"]
    for review in d["review"]:
        terms += review["review/text"]
    doc_terms.append(terms)

tf_idf = vectorizer.fit_transform([d for d in doc_terms])

docs_compressed, s, words_compressed = svds(tf_idf, k=150)

num_splits = 10
split = np.split(words_compressed, num_splits)
for k in range(num_splits):
    # save as json
    np.save("words_compressed_" + str(k), split[k])

np.save("docs_compressed", docs_compressed @ np.diag(s))
np.save("s", s)
import matplotlib.pyplot as plt

plt.plot(s[::-1])
plt.show()

import joblib

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
