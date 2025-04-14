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

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, "init.json")

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, "r") as file:
    data = json.load(file)

vectorizer = TfidfVectorizer(use_idf=True, strip_accents="unicode")
doc_terms = []
for d in data:
    terms = d["description"]
    for review in d["review"]:
        terms += review["review/text"]
    doc_terms.append(terms)

tf_idf = vectorizer.fit_transform([d for d in doc_terms])

docs_compressed, s, words_compressed = svds(tf_idf, k=75)
np.save("docs_compressed", docs_compressed)
np.save("words_compressed", words_compressed)
import matplotlib.pyplot as plt

plt.plot(s[::-1])
plt.show()

import joblib

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
