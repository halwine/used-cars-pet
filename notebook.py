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
    return


@app.cell
def _(raw_df):
    raw_df.nunique()
    return


@app.cell
def _(raw_df):
    # Check missing values percent for each attribute
    raw_df.isna().mean() * 100
    return


@app.cell
def _():
    """
    Before data cleaning: 
        state, odometer and price - 0%  
    
        size - 72% missing values and should be deleted
    
        condition - 38%
        cylinders - 41%
        drive - 30%
        type - 22%
        paint_color - 29%

        year - 0.3%
        fuel - 0.6%
        title_status - 1.8%
        transmission - 0.4%

            Irrelevant data that should be dropped:
        id
        url
        region_url      
        image_url
        posting_date
        VIN

            Useless data as we have "state" geo attribute
        region
        lat
        long

            0 Indices
        county

            Noisy data
        model
        description
    
    """
    return


@app.cell
def _(raw_df):
    # Save original df
    original_df = raw_df.copy()

    # Drop irrelevant features
    df = raw_df.drop([
        # Irrelevant data:
        "id",    
        "url",    
        "region_url",    
        "image_url",    
        "posting_date",    

        # Does not affect car's value
        "VIN",    

        # Irrelevant data as we have "state" geo attribute
        "region",
        "lat",    
        "long",

        # Noisy data
        "model",
        "description",
    
    
        "size",    # 70% missing values
        "county"    # Have 0 indices
    ], axis=1)

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

    # List of columns with little missing data
    attribs_to_clean = ["year", "fuel", "transmission", "manufacturer", "title_status"]

    practice_df = practice_df.dropna(subset=attribs_to_clean)

    """
    Next step is to clear data from:

    year (0.2850507449%)
    fuel (0.6134003372%)
    transmission (0.4041225751%)

    title_status (1.7663282363%)
    manufacturer (3.9079126572%)

    summary data loss: 19396 indices
    """
    from sklearn.impute import SimpleImputer

    # List of columns with more missing data (21% - 40%)
    attribs_to_clean = ["condition", "cylinders", "drive",
                       "type", "paint_color", ""]


    # 1. Condition: fill gaps with "unknown"
    imputer = SimpleImputer(strategy="constant", fill_value="unknown")

    practice_df[["condition"]] = imputer.fit_transform(practice_df[["condition"]])


    # 2. Type: fill gaps with "unknown"
    practice_df[["type"]] = imputer.fit_transform(practice_df[["type"]])


    # 3. Drive: fill NaN(s) depending on average car's type 
    # If type is unknown, remain drive unknown aswell

    converter_dict = {
        "unknown": "unknown",
        "SUV": "4wd",
        "bus": "rwd",
        "convertible": "rwd",
        "coupe": "rwd",
        "hatchback": "fwd",
        "sedan": "fwd",
        "truck": "4wd",
        "van": "fwd",
        "wagon": "4wd",
    }

    drive_from_type = practice_df['type'].map(converter_dict)
    practice_df['drive'] = practice_df['drive'].fillna(drive_from_type)

    # Mapping made missing values drop from 40% to 5% total
    # Drop remaining 5% indices with missing values
    practice_df = practice_df.dropna(subset="drive")



    return (practice_df,)


@app.cell
def _(practice_df):
    # Dataset after initial cleaning
    practice_df.isna().mean() * 100
    return


if __name__ == "__main__":
    app.run()
