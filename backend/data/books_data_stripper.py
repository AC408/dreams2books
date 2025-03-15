import pandas as pd

df = pd.read_csv("books_data.csv")
df = df.drop(columns=["image", "previewLink", "publisher", "publishedDate"])

df.to_csv("books_data_stripped.csv", index=False)
