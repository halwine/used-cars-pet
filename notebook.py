import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    return np, pd


@app.cell
def _(pd):
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("austinreese/craigslist-carstrucks-data")

    print("Path to dataset files:", path)
    raw_df = pd.read_csv(path + "/" + "vehicles.csv")
    return (raw_df,)


@app.cell
def _(raw_df):
    raw_df.head()
    return


@app.cell
def _(raw_df):
    raw_df.info()

    # There are many attributes with missing features.
    return


@app.cell
def _(raw_df):
    raw_df.nunique()
    return


@app.cell
def _(raw_df):
    # Save original df
    original_df = raw_df.copy()

    # Drop irrelevant features
    df = raw_df.drop(["id", "url", "region_url", "image_url", "county", "VIN"], axis=1)

    # Drop cars with price less than 500 and more than 200_000
    df = df[df["price"].between(500, 200_000)]

    # Drop cars with odometer more than 400_000
    df = df[df["odometer"] < 400_000]

    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df, np, pd):
    # Initialize stratified attribute
    df["price_cat"] = pd.cut(df["price"],
                            bins=[0., 7000., 15000., 25000., 45000., np.inf],
                            labels=[1, 2, 3, 4, 5])
    return


@app.cell
def _(df):
    df["price_cat"].hist()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
