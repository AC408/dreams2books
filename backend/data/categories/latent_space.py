import os
import numpy as np
from sklearn.preprocessing import normalize
import json
import gc

NUM_INDICES_IN_EACH_DATA_FILE = 11309  # determined by running create_init_json

root_dir = os.path.abspath(os.path.join("..", os.curdir))

docs_compressed_normed = normalize(np.load(root_dir + "/data/svd/docs_compressed.npy"))

# SVD gives eigenvalues (dimensions) in ascending order, we want the most variabilities
U_30 = docs_compressed_normed[:, -30:]
print(U_30.shape)  # double checking shape

for i in range(30):
    # Get top 30 documents most related to each dimension
    top_docs = U_30[:, i].argsort()[::-1][:30]

    matched_res = []
    for j in range(30):
        index = top_docs[j]
        file_num = int(index / NUM_INDICES_IN_EACH_DATA_FILE)
        # here, we should load the relevant data
        with open(
            root_dir + "/data/book_info/data_" + str(file_num) + ".json", "r"
        ) as file:
            data = json.load(file)
        index_in_file = index - NUM_INDICES_IN_EACH_DATA_FILE * file_num
        # then grab the data and append
        matched_res.append(data[index_in_file])

    # Write to a JSON file
    with open(
        root_dir + "/data/categories/latent_dimensions" + str(i) + ".json", "w"
    ) as f:
        json.dump(matched_res, f, indent=4)  # indent makes it pretty
