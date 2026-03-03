import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    return (pd,)


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
    raw_df.describe()
    return


@app.cell
def _(raw_df):
    # Save original df
    original_df = raw_df.copy()

    # Drop irrelevant features
    df = raw_df.drop(["id", "url", "region_url", "image_url", "county", "VIN"], axis=1)


    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
