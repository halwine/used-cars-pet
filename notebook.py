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
    raw_df.info()
    return


@app.cell
def _(raw_df):
    raw_df.nunique()
    return


@app.cell
def _(raw_df):
    # Check missing values percent for each attribute
    raw_df.isna().mean().sort_values(ascending=False) * 100
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
def _(pd, raw_df):
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


    # Convert 'cylinders' column to int

        # Get rid of ' cylinders'
    tmp_cyl = df['cylinders'].str.replace(' cylinders', '', regex=False)    

        # Replace e.g. "6 cylinders" with just 6
        # Convert 'other' to NaN
    df['cylinders'] = pd.to_numeric(tmp_cyl, errors='coerce').astype('Int64')    


    # Reset indexes of deleted instancies
    df = df.reset_index(drop=True)
    return (df,)


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
    # Stratified split to evenly destribute cars with different prices among test and train set
    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.05,
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


    # ---------- DATA CLEANING ----------

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

    # 3. Paint_color: fill gaps with "unknown"
    practice_df[["paint_color"]] = imputer.fit_transform(practice_df[["paint_color"]])

    # 4. Drive: fill NaN(s) depending on average car's type 
    # If type is unknown, remain drive unknown aswell

    drive_modes_by_type = {
        "unknown": "unknown",
        "SUV": "4wd",
        "bus": "rwd",
        "convertible": "rwd",
        "coupe": "rwd",
        "hatchback": "fwd",
        "mini-van": "fwd",
        "offroad": "4wd",
        "other": "fwd",
        "pickup": "fwd",
        "sedan": "fwd",
        "truck": "4wd",
        "van": "fwd",
        "wagon": "4wd",
    }

    drive_from_type = practice_df['type'].map(drive_modes_by_type)
    practice_df['drive'] = practice_df['drive'].fillna(drive_from_type)

    # 5. Cylinders: Fill NaN's depending on average cylinders-per-type value

    cyl_modes_by_type = {
        "unknown": 6,
        "SUV": 6,
        "bus": 8,
        "convertible": 8,
        "coupe": 8,
        "hatchback": 4,
        "mini-van": 6,
        "offroad": 6,

        "other": 6,
        "pickup": 8,
        "sedan": 4,
        "truck": 8,
        "van": 6,
        "wagon": 4
    }

    cyls_from_type = practice_df['type'].map(cyl_modes_by_type)
    practice_df['cylinders'] = practice_df['cylinders'].fillna(cyls_from_type)


    # ---------- FEATURE ENGINEERING ----------

    # Check corr matrix
    # print(practice_df.corr(numeric_only=True))

    # Create 'car_age' attribute
    practice_df['car_age'] = 2026 - practice_df['year']

    # Create 'miles_per_year' attribute
    # Adding 1 to avoid zero division if there's a car made in 2026 in df
    practice_df['miles_per_year'] = practice_df['odometer'] / (practice_df['car_age'] + 1)

    # Drop 'year' column as it has -1 corr to 'car_age' column 
    practice_df = practice_df.drop('year', axis=1)

    # Corr matrix after clearing NaNs and feature engineering
    # print(practice_df.corr(numeric_only=True))


    # ---------- OneHot encoded columns ----------
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

    numeric_cats = ['odometer', 'cylinders', 'car_age', 'miles_per_year']
    onehot_cats = ['transmission', 'drive', 'fuel', 'paint_color', 'type']
    ordinal_cats = ['title_status', 'condition']

    title_status_order = ['missing', 'parts_only', 'salvage', 'rebuilt', 'lien', 'clean']
    condition_order = ['salvage', 'fair', 'good', 'excellent','like new', 'new']

    # ct = ColumnTransformer[
    #     () 
    # ]

    # cat_encoder = OneHotEncoder(sparse_output=False)
    # transmission_cat_1hot = cat_encoder.fit_transform(practice_df[['transmission'])]
    return


if __name__ == "__main__":
    app.run()
