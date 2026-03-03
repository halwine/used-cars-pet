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
    df = raw_df.drop(["id", "url", "region_url", "image_url", "county", "VIN", "posting_date"], axis=1)

    # Drop cars with price less than 500 and more than 200_000
    df = df[df["price"].between(500, 200_000)]

    # Drop cars with odometer more than 400_000
    df = df[df["odometer"] < 400_000]

    # Reset indexes of deleted instancies
    df = df.reset_index(drop=True)
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df, np, pd):
    # Initialize stratified price attribute
    df["price_cat"] = pd.cut(df["price"],
                            bins=[0., 7000., 15000., 25000., 45000., np.inf],
                            labels=[1, 2, 3, 4, 5])
    return


@app.cell
def _(df):
    df["price_cat"].hist()
    return


@app.cell
def _(df):
    # Stratified split to evenly destribute cars with different prices amond test and train set
    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                  random_state=42)

    for train_index, test_index in split.split(df, df["price_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    # Delete price_cat attribute 
    strat_train_set = strat_train_set.drop("price_cat", axis=1)
    strat_test_set = strat_test_set.drop("price_cat", axis=1)
    return (strat_train_set,)


@app.cell
def _(strat_train_set):
    # Create copy of train set for practice
    practice_df = strat_train_set.copy()
    return (practice_df,)


@app.cell
def _(practice_df):
    practice_df.info()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
