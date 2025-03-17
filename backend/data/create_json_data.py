import pandas as pd
import json

df_books_data = pd.read_csv("books_data_stripped.csv")

# remove books without titles
df_books_data = df_books_data[df_books_data["Title"].notna()]

# make titles lowercase and without extra quotations
df_books_data["Title"] = df_books_data["Title"].apply(
    lambda x: x.replace("'", "").replace('"', "").lower()
)
df_books_data = df_books_data.sort_values(by=["Title"])

# remove any books with titles that are subset of other books
# for example, "A is for Alibi" because there's "A is for Alibi (SIGNED)"
# otherwise, we have duplicate data
parsed_book = []
for index in range(len(df_books_data)):
    row = df_books_data.iloc[index]
    above = False
    below = False
    if index > 0 and row["Title"] in df_books_data.iloc[index - 1]["Title"]:
        above = True
    if (
        index < len(df_books_data) - 1
        and row["Title"] in df_books_data.iloc[index + 1]["Title"]
    ):
        below = True
    if not above and not below:
        parsed_book.append(row)
df_books_data = pd.DataFrame(parsed_book)

df_books_rating = pd.read_csv("books_rating_stripped.csv")

# remove books without titles
df_books_rating = df_books_rating[df_books_rating["Title"].notna()]

# make titles lowercase and without extra quotations
df_books_rating["Title"] = df_books_rating["Title"].apply(
    lambda x: x.replace("'", "").replace('"', "").lower()
)

# alternative method to removing duplicate titles, but some titles have more reviews in one title than another
# for example, "A is for Alibi (SIGNED)" has 1 extra review over "A is for Alibi"
# df_books_rating = df_books_rating.drop_duplicates(subset=["review/text"])

# merge
df = df_books_data.merge(df_books_rating, on="Title", how="inner")

# merge all reviews per book under one label
# this means each row has a unique book
# otherwise, each row can refer to the same book
df = (
    df.groupby(
        ["Title", "description", "authors", "infoLink", "categories", "ratingsCount"],
        dropna=False,
    )
    .apply(lambda x: x[["review/helpfulness", "review/text"]].to_dict(orient="records"))
    .reset_index(name="reviews")
)

# turn to dictionary to store as json
df_dic = df.to_dict(orient="records")

# save as json
with open("books_data_json.json", "w") as json_file:
    json.dump(df_dic, json_file, indent=4)
