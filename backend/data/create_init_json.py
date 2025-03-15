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

# turn to dictionary to store as json
df_dic = df_books_data.to_dict(orient="records")

# save as json
with open("init.json", "w") as json_file:
    json.dump(df_dic, json_file, indent=4)
