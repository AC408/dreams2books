import pandas as pd
import json
import numpy as np
import re

MIN_NUM_REVIEW = 3  # Each books need MIN_NUM_REVIEW reviews
MIN_DESCRIPTION_CHAR = 20  # Each book descriptions need MIN_DESCRIPTION_CHAR chars
MIN_REVIEW_HELPFULNESS = (
    0.75  # Each review has to be at least MIN_REVIEW_HELPFULNESS helpful
)
MIN_NUM_REVIEW_OF_REVIEW = (
    3  # Each review needs to have MIN_NUM_REVIEW_OF_REVIEW reviews
)
MIN_REVIEW_CHAR = 20  # Each review needs MIN_REVIEW_CHAR chars
MAX_REVIEW_COUNT_PER_BOOK = 5  # Max number of reviews per book to conserve space

path_to_books_data = "~/Downloads/archive/books_data.csv"

df_books_data = pd.read_csv(path_to_books_data)
df_books_data = df_books_data.drop(
    columns=["image", "previewLink", "publisher", "publishedDate"]
)

# remove books without titles
df_books_data = df_books_data[df_books_data["Title"].notna()]

# remove books without descriptions
df_books_data = df_books_data[df_books_data["description"].notna()]

# make titles lowercase and without punctuations
df_books_data["Title"] = df_books_data["Title"].apply(
    lambda x: re.sub(r"[^\w\s]", "", x).lower()
)

df_books_data = df_books_data.drop_duplicates()

df_books_data = df_books_data.sort_values(by=["Title"])

asci_mapping = {
    "“": '"',
    "”": '"',
    "’": "'",
    "®": "",
    "—": "-",
    "–": ":",
    "…": "...",
    "•": "->",
    "●": "->",
    "×": "x",  # non-ascii
    "–": "-",
    "©": "",
    "°": "",
    "  ": " ",
}

# remove any books with titles that are subset of other books
# for example, "A is for Alibi" because there's "A is for Alibi (SIGNED)"
# otherwise, we have duplicate data
parsed_book_data = {}
for index in range(len(df_books_data)):
    row = df_books_data.iloc[index]

    # skip books with descriptions length < 20
    if len(row["description"]) < MIN_DESCRIPTION_CHAR:
        continue
    # convert english non-ascii to ascii
    new_description = row["description"]
    new_title = row["Title"]
    for k in asci_mapping.keys():
        new_description = new_description.replace(k, asci_mapping[k])
        new_title = new_title.replace(k, asci_mapping[k])
    new_description = new_description.strip()
    new_title = new_title.strip()
    df_books_data.iat[index, df_books_data.columns.get_loc("description")] = (
        new_description
    )
    df_books_data.iat[index, df_books_data.columns.get_loc("Title")] = new_title
    row = df_books_data.iloc[index]

    # keep books with english descriptions and titles
    if not row["description"].isascii():
        continue
    if not row["Title"].isascii():
        continue

    # Skip if the current title is a substring of the previous title and they have the same description and authors
    if index > 0:
        prev_row = df_books_data.iloc[index - 1]
        prev_title = prev_row["Title"]
        prev_description = prev_row["description"]
        prev_authors = prev_row["authors"]

        # Check if this title is a substring of the previous title or vice versa, and descriptions and authors match
        if (
            (row["Title"] in prev_title or prev_title in row["Title"])
            and prev_description == row["description"]
            and prev_authors == row["authors"]
        ):

            continue

    d = {
        "Title": row["Title"],
        "description": row["description"],
        "authors": row["authors"],
        "infoLink": row["infoLink"],
        "categories": row["categories"],
        "ratingsCount": row["ratingsCount"],
    }
    parsed_book_data[row["Title"]] = d

path_to_books_rating = "~/Downloads/archive/Books_rating.csv"

df_books_rating = pd.read_csv(path_to_books_rating)
df_books_rating = df_books_rating.drop(
    columns=[
        "Id",
        "Price",
        "User_id",
        "profileName",
        "review/score",
        "review/time",
        "review/summary",
    ]
)

# remove books without titles
df_books_rating = df_books_rating[df_books_rating["Title"].notna()]

# remove books without reviews
df_books_rating = df_books_rating[df_books_rating["review/text"].notna()]

# make titles lowercase and without extra quotations
df_books_rating["Title"] = df_books_rating["Title"].apply(
    lambda x: re.sub(r"[^\w\s]", "", x).lower()
)

df_books_rating = df_books_rating.drop_duplicates()

df_books_rating = df_books_rating.sort_values(by=["Title"])

parsed_books = []
row = df_books_rating.iloc[0]
title = row["Title"]
reviews = []  # starts empty because loop starts at row 0
index = 0
while index < len(df_books_rating):
    row = df_books_rating.iloc[index]
    if row["Title"] == title:  # combine reviews together
        review_helpfulness = row["review/helpfulness"]
        index_of_slash = review_helpfulness.index("/")
        denominator = int(review_helpfulness[index_of_slash + 1 :])
        if (
            denominator < MIN_NUM_REVIEW_OF_REVIEW
        ):  # skips reviews with less than x reviews
            index += 1
            continue
        numerator = int(review_helpfulness[:index_of_slash])
        if (
            numerator / denominator <= MIN_REVIEW_HELPFULNESS
        ):  # skips reviews with helpfulness less than x
            index += 1
            continue
        new_review = row["review/text"]
        for k in asci_mapping.keys():
            new_review = new_review.replace(k, asci_mapping[k])
        new_review = new_review.strip()
        if len(new_review) < MIN_REVIEW_CHAR:  # skips reviews with less than x chars
            index += 1
            continue
        df_books_rating.iat[index, df_books_rating.columns.get_loc("review/text")] = (
            new_review
        )
        row = df_books_rating.iloc[index]
        if not row["review/text"].isascii():  # remove non-english reviews
            index += 1
            continue
        review = {
            "review/text": row["review/text"],
            "review/helpfulness": row["review/helpfulness"],
            "num_reviews": denominator,
        }
        reviews.append(review)  # combine reviews
        index += 1
    else:
        for k in asci_mapping.keys():
            title = title.replace(k, asci_mapping[k])
        title = title.strip()
        if (
            title.isascii() and title in parsed_book_data
        ):  # removes non-english review, performs inner merge
            if (
                len(reviews) >= MIN_NUM_REVIEW
            ):  # only include books with some number of reviews
                data = parsed_book_data[title]
                sorted_reviews = sorted(
                    reviews, key=lambda x: x["num_reviews"], reverse=True
                )
                data["review"] = sorted_reviews[:MAX_REVIEW_COUNT_PER_BOOK]
                parsed_books.append(data)
        title = row["Title"]
        reviews = []
df_books_data = pd.DataFrame(parsed_books)

# turn to dictionary to store as json
df_dic = df_books_data.to_dict(orient="records")
num_dictionary = len(df_dic)
num_splits = 5
num_steps = int(num_dictionary / num_splits) + 1
print(num_steps)
start = 0
for k in range(num_splits):
    # save as json
    with open("data_" + str(k) + ".json", "w") as json_file:
        json.dump(
            df_dic[start : min(num_dictionary, start + num_steps)],
            json_file,
            indent=4,
        )
    start += num_steps
