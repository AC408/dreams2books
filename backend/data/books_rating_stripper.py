import pandas as pd

df = pd.read_csv("Books_rating.csv")
df = df.drop(
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

df.to_csv("books_rating_stripped.csv", index=False)
