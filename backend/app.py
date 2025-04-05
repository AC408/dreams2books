import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

# from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import numpy as np

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

tf_idf = vectorizer.fit_transform([d for d in doc_terms]).toarray()
index_to_vocab = {i: v for i, v in enumerate(vectorizer.get_feature_names_out())}
vocab_to_index = {}
for k in index_to_vocab.keys():
    vocab_to_index[index_to_vocab[k]] = k

app = Flask(__name__)
CORS(app)


# Sample search using json with pandas
def json_search(query):
    # TODO: CHANGE THIS TO USE TF-IDF AND THEN USE SVD
    query = query.lower()
    query_tokens = vectorizer.build_tokenizer()(query)

    phrases = []
    max_words = min(10, len(query_tokens))
    for i in range(max_words - 1):
        phrases.append(query_tokens[i] + " " + query_tokens[i + 1])
    for i in range(max_words - 2):
        phrases.append(
            query_tokens[i] + " " + query_tokens[i + 1] + " " + query_tokens[i + 2]
        )

    doc_scores = {}
    q_freq = {}

    # count word frequencies, but focus on meaningful words
    for word in query_tokens:
        # skip very short words
        if len(word) > 2:
            if word in q_freq:
                q_freq[word] += 1
            else:
                q_freq[word] = 1

    q_vec = np.zeros(tf_idf.shape[1])
    for token in q_freq:
        count = q_freq[token]
        if token in index_to_vocab.values():
            word_idf = vectorizer.idf_[vectorizer.vocabulary_[token]]
            q_tfidf = count * word_idf
            q_vec[vocab_to_index[token]] = q_tfidf

    cosine_similarity = np.dot(tf_idf, q_vec)

    for doc in range(tf_idf.shape[0]):
        # more weight for the titles
        title_bonus = 1.0
        if token in data[doc]["Title"]:
            title_bonus = 3.0

        phrase_bonus = 1.0
        for phrase in phrases:
            if phrase in data[doc]["description"]:
                phrase_bonus = 2.0
                break

        cosine_similarity[doc] = (
            cosine_similarity[doc]
            * title_bonus
            * phrase_bonus
            / np.linalg.norm(tf_idf[doc])
        )

    top_ten_score = np.sort(cosine_similarity)[::-1][:10]
    top_ten_docs = np.argsort(cosine_similarity)[::-1][:10]

    matched_res = []
    for i in range(10):
        matched_res.append(data[top_ten_docs[i]])

    df = pd.DataFrame(matched_res)
    df["similarity_score"] = top_ten_score
    # TODO: Add photos, hyperlink, ISSN
    return df.to_json(orient="records")


@app.route("/")  # calls render_template when url accessed
def home():
    return render_template("base.html", title="sample html")


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
