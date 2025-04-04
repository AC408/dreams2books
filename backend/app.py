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

vectorizer = TfidfVectorizer(strip_accents="unicode")
data_vector = []
for d in data:
    if isinstance(d["description"], str):
        data_vector.append(d["description"])
    else:
        data_vector.append("")

data_tokenizer = lambda d_v: [vectorizer.build_tokenizer()(d.lower()) for d in d_v]
data_tokens = data_tokenizer(data_vector)
# data_tokens = [TreebankWordTokenizer.tokenize(d.lower()) for d in data_vector]
num_books = len(data_tokens)
inverted_index = {}
for i in range(num_books):
    toks = data_tokens[i]
    for tok in toks:
        if tok in inverted_index:
            ids = inverted_index[tok]
            last_id = len(ids) - 1
            last_tup = ids[last_id]
            if last_tup[0] == i:
                ids[last_id] = (i, last_tup[1] + 1)
            else:
                inverted_index[tok] += [(i, 1)]
        else:
            inverted_index[tok] = [(i, 1)]

idf = {}
for key in inverted_index.keys():
    val = inverted_index[key]
    length = len(val)
    df_ratio = length / num_books
    if df_ratio < 0.5:
        idf[key] = math.log2(num_books / (1 + length))

norms = np.zeros((num_books))
for word in inverted_index.keys():
    for docs, tf in inverted_index[word]:
        norms[docs] += (tf * idf.get(word, 0)) ** 2

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
    q_norm = 0
    q_freq = {}

    # count word frequencies, but focus on meaningful words
    for word in query_tokens:
        # skip common words and very short words
        if len(word) > 2:
            if word in q_freq:
                q_freq[word] += 1
            else:
                q_freq[word] = 1

    for token in q_freq:
        count = q_freq[token]
        if token in inverted_index and token in idf:
            q_norm += (q_freq[token] * idf[token]) ** 2
            ids = inverted_index[token]
            for doc, tf in ids:
                word_idf = idf[token]

                # more weight for the titles
                title_bonus = 1.0
                if token in data[doc]["Title"].lower():
                    title_bonus = 3.0

                phrase_bonus = 1.0
                for phrase in phrases:
                    if phrase in data[doc]["description"].lower():
                        phrase_bonus = 2.0
                        break

                score = count * word_idf * tf * word_idf * title_bonus * phrase_bonus

                if doc in doc_scores:
                    doc_scores[doc] += score
                else:
                    doc_scores[doc] = score

    q_norm = q_norm ** (0.5)
    results = []
    similarity_score = []
    for i in range(num_books):
        if i in doc_scores:
            norm = norms[i]
            numerator = doc_scores[i]
            results.append((numerator / (norm * q_norm), i))
            similarity_score.append(numerator / (norm * q_norm))
        else:
            results.append((0, i))
            similarity_score.append(0)
    sorted_res = sorted(results, key=lambda x: x[0], reverse=True)
    sorted_similarity_score = sorted(similarity_score, reverse=True)

    matched_res = []
    for i in range(10):
        matched_res.append(data[sorted_res[i][1]])

    df = pd.DataFrame(matched_res)
    df["similarity_score"] = sorted_similarity_score[:10]
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
