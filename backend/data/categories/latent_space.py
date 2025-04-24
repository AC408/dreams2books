import os
import numpy as np
from sklearn.preprocessing import normalize
import json
import gc

NUM_INDICES_IN_EACH_DATA_FILE = 11450  # determined by running create_init_json

root_dir = os.path.abspath(os.path.join("..", os.curdir))
svd_path = root_dir + "/svd/"

docs_compressed_normed = normalize(np.load(svd_path + "docs_compressed.npy"))

U_30 = docs_compressed_normed[
    :, -30:
]  # SVD gives eigenvalues (dimensions) in ascending order, we want the most variabilities
print(U_30.shape)
for i in range(30):
    top_docs = U_30[:, i].argsort()[::-1][
        :30
    ]  # Get top 30 documents most related to each dimensions
    print(f"Top documents for dimension {i}: {top_docs[:5]}")
    matched_res = []
    for j in range(30):
        index = top_docs[j]
        file_num = int(index / NUM_INDICES_IN_EACH_DATA_FILE)
        # here, we should load the relevant data
        with open(root_dir + "/book_info/data_" + str(file_num) + ".json", "r") as file:
            data = json.load(file)
        index_in_file = index - NUM_INDICES_IN_EACH_DATA_FILE * file_num
        # then grab the data and append
        matched_res.append(data[index_in_file])
        # then delete the resource
        del data
        # then have garbage collector collect it
        gc.collect()

    # Write to a JSON file
    with open("latent_dimensions" + str(i) + ".json", "w") as f:
        json.dump(matched_res, f, indent=4)  # indent makes it pretty
