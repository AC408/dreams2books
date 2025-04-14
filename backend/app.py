import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from scipy.sparse.linalg import svds

# from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import numpy as np
from sklearn.preprocessing import normalize

import joblib

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

vectorizer = joblib.load("tfidf_vectorizer.pkl")
words_compressed = np.load("words_compressed.npy")
docs_compressed_normed = normalize(np.load("docs_compressed.npy"))

app = Flask(__name__)
CORS(app)


# Sample search using json with pandas
def json_search(query):
    query = query.lower()
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words_compressed.T)).squeeze()
    sims = docs_compressed_normed.dot(query_vec)

    scores = np.sort(sims)[::-1][:10]
    asort = np.argsort(sims)[::-1][:10]
    matched_res = []
    for i in range(10):
        matched_res.append(data[asort[i]])

    df = pd.DataFrame(matched_res)
    df["similarity_score"] = scores
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
    app.run(debug=True, host="0.0.0.0", port=5001)
