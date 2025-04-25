import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd

import numpy as np
from sklearn.preprocessing import normalize

import gc
import joblib

NUM_INDICES_IN_EACH_DATA_FILE = 11309  # determined by running create_init_json
MAX_NUM_RESULTS = 50
NUM_FEATURES = 30
TOP_K_FEATURES = 3

# Feature list corresponding to the last NUM_FEATURES dimensions
FEATURE_LIST = ["History", "International", "Games", "Education", "SocialScience", "Performance", "Faith", "Entertainment", "Pets", "Care", "Literature", "Art", "Verse", "Culture", "Vacation", "Identity", "Narrative", "Technology", "Linguistics", "Spirituality", "Classics", "Regional", "Theology", "Fiction", "Conflict", "Crime", "Idioms", "Cuisine", "Suspense", "Reference"]

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
root_dir = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
svd_path = current_directory + "/data/svd/"

# IF WE NEED MORE MEMORY, LOAD ON DEMAND AND IMMEDIATELY DELETE AFTERWARD
vectorizer = joblib.load(svd_path + "tfidf_vectorizer.pkl")
words_compressed = np.load(svd_path + "words_compressed_0.npy")
num_split = 5  # same as number of word_compressed files and in save_svd_data
for k in range(1, num_split):
    new_words_compressed = np.load(svd_path + "words_compressed_" + str(k) + ".npy")
    words_compressed = np.concatenate((words_compressed, new_words_compressed))
docs_compressed = np.load(svd_path + "docs_compressed.npy")
docs_compressed_normed = normalize(docs_compressed)
index_to_term = vectorizer.get_feature_names_out()

app = Flask(__name__)
CORS(app)


# Sample search using json with pandas
def json_search(query):
    query = query.lower()
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words_compressed.T)).squeeze()
    sims = docs_compressed_normed.dot(query_vec)

    scores = np.sort(sims)[::-1][:MAX_NUM_RESULTS]
    asort = np.argsort(sims)[::-1][:MAX_NUM_RESULTS]
    matched_res = []
    cos_explanation = []
    hidden_cos_explanation = []
    top_features_list = [] # New list to store top features for each book

    for i in range(MAX_NUM_RESULTS):
        index = asort[i]
        file_num = int(index / NUM_INDICES_IN_EACH_DATA_FILE)
        # here, we should load the relevant data
        with open(
            current_directory + "/data/book_info/data_" + str(file_num) + ".json", "r"
        ) as file:
            data = json.load(file)
        index_in_file = index - NUM_INDICES_IN_EACH_DATA_FILE * file_num
        # then grab the data and append
        result = data[index_in_file]
        matched_res.append(result)

        # Calculate top features based on the book's vector
        book_vector = docs_compressed_normed[index]
        feature_scores = book_vector[-NUM_FEATURES:]
        top_feature_indices = np.argsort(feature_scores)[::-1][:TOP_K_FEATURES]
        top_features = [FEATURE_LIST[idx] for idx in top_feature_indices]
        top_features_list.append(top_features)

        # emphasize words corresponding to query
        result_q = result["Title"] + result["description"].lower()
        result_v = vectorizer.transform([result_q]).toarray()
        cos_res = np.array(result_v.squeeze() * query_tfidf.squeeze())
        max_sim = min(np.count_nonzero(cos_res), 10)
        sim_terms = np.argsort(cos_res)[::-1][:max_sim]
        cos_explanation_i = []
        for t in range(len(sim_terms)):
            term_index = sim_terms[t]
            term = index_to_term[term_index]
            cos_explanation_i.append(term)
        cos_explanation.append(cos_explanation_i)

        hidden_q = ""
        for review in result["review"]:
            hidden_q += review["review/text"].lower()
        hidden_v = vectorizer.transform([hidden_q]).toarray()
        hidden_cos_res = np.array(hidden_v.squeeze() * query_tfidf.squeeze())
        hidden_max_sim = min(np.count_nonzero(hidden_cos_res), 10)
        hidden_sim_terms = np.argsort(hidden_cos_res)[::-1][:hidden_max_sim]
        hidden_cos_explanation_i = []
        for t in range(len(hidden_sim_terms)):
            hidden_term_index = hidden_sim_terms[t]
            hidden_term = index_to_term[hidden_term_index]
            hidden_cos_explanation_i.append(hidden_term)
        hidden_cos_explanation.append(hidden_cos_explanation_i)

        # then delete the resource
        del data
        # then have garbage collector collect it
        gc.collect()

    df = pd.DataFrame(matched_res)
    df["similarity_score"] = scores
    df["cos_explanation"] = cos_explanation
    df["hidden_explanation"] = hidden_cos_explanation
    df["top_features"] = top_features_list # Add top features to DataFrame
    print(hidden_cos_explanation)
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
