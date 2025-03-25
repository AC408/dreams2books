import pandas as pd
import json

df_books_data = pd.read_csv("data/books_data_stripped.csv")

# remove books without titles
df_books_data = df_books_data[df_books_data["Title"].notna()]

# remove books without descriptions
df_books_data = df_books_data[df_books_data["description"].notna()]

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
    if above or below:
        continue
    # skip books with descriptions length < 20
    if len(row["description"]) < 20:
        continue
    # convert english non-ascii to ascii
    if (
        "“" in row["description"]
        or "”" in row["description"]
        or "’" in row["description"]
        or "®" in row["description"]
        or "–" in row["description"]
        or "—" in row["description"]
        or "…" in row["description"]
        or "•" in row["description"]
        or "●" in row["description"]
        or "×" in row["description"]  # non-ascii
        or "“" in row["Title"]
        or "”" in row["Title"]
        or "’" in row["Title"]
        or "®" in row["Title"]
        or "—" in row["Title"]
        or "…" in row["Title"]
        or "•" in row["Title"]
        or "●" in row["Title"]
        or "×" in row["Title"]
        or "–" in row["Title"]
    ):
        df_books_data.iat[index, df_books_data.columns.get_loc("description")] = (
            row["description"]
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("®", "")
            .replace("—", "-")
            .replace("…", "...")
            .replace("•", "->")
            .replace("●", "->")
            .replace("×", "x")
            .replace("–", "-")
        )
        df_books_data.iat[index, df_books_data.columns.get_loc("Title")] = (
            row["Title"]
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("®", "")
            .replace("—", "-")
            .replace("…", "...")
            .replace("•", "->")
            .replace("●", "->")
            .replace("×", "x")
            .replace("–", "-")
        )
    row = df_books_data.iloc[index]
    # # keep books with english descriptions and titles
    # if not row["description"].isascii():
    #     continue
    # if not row["Title"].isascii():
    #     continue
    parsed_book.append(row)
df_books_data = pd.DataFrame(parsed_book)
# remove books with the same descriptions or titles
# df_books_data = df_books_data.drop_duplicates(subset=["description", "Title"])

# maybe remove books with less than x reviews, same reviews, reviews with less than x%, reviews not in english?

# turn to dictionary to store as json
df_dic = df_books_data.to_dict(orient="records")

# save as json
with open("init.json", "w") as json_file:
    json.dump(df_dic, json_file, indent=4)
