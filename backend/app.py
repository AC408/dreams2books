import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd

import numpy as np
from sklearn.preprocessing import normalize

import gc
import joblib

NUM_INDICES_IN_EACH_DATA_FILE = 3994  # determined by running create_init_json
MAX_NUM_RESULTS = 50

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))


def get_path(file_name):
    return os.path.join(current_directory, file_name)


# IF WE NEED MORE MEMORY, LOAD ON DEMAND AND IMMEDIATELY DELETE AFTERWARD
vectorizer = joblib.load(get_path("tfidf_vectorizer.pkl"))
words_compressed = np.load(get_path("words_compressed_0.npy"))
num_split = 5
for k in range(1, num_split):
    new_words_compressed = np.load(get_path("words_compressed_" + str(k) + ".npy"))
    words_compressed = np.concatenate((words_compressed, new_words_compressed))
docs_compressed_normed = normalize(np.load(get_path("docs_compressed.npy")))
s = np.load(get_path("s.npy"))

app = Flask(__name__)
CORS(app)


# Sample search using json with pandas
def json_search(query):
    query = query.lower()
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words_compressed.T / s)).squeeze()
    sims = docs_compressed_normed.dot(query_vec)

    num_results = min(np.count_nonzero(sims), MAX_NUM_RESULTS)
    scores = np.sort(sims)[::-1][:num_results]
    asort = np.argsort(sims)[::-1][:num_results]
    matched_res = []
    for i in range(num_results):
        index = asort[i]
        file_num = int(index / NUM_INDICES_IN_EACH_DATA_FILE)
        # here, we should load the relevant data
        with open(get_path("data_" + str(file_num) + ".json"), "r") as file:
            data = json.load(file)
        index_in_file = index - NUM_INDICES_IN_EACH_DATA_FILE * file_num
        # then grab the data and append
        matched_res.append(data[index_in_file])
        # then delete the resource
        del data
        # then have garbage collector collect it
        gc.collect()

    df = pd.DataFrame(matched_res)
    df["similarity_score"] = scores
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
