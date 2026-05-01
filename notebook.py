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
    df['cylinders'] = pd.to_numeric(tmp_cyl, errors='coerce').astype('float64')    


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

    # Initial data cleaning 
    attribs_to_clean = ["year", "fuel", "transmission", "manufacturer", "title_status"]

    strat_train_set = strat_train_set.dropna(subset=attribs_to_clean)
    strat_test_set = strat_test_set.dropna(subset=attribs_to_clean)
    return strat_test_set, strat_train_set


@app.cell
def _():
    # ---------- LISTS AND DICTS ----------

    from CustomImputers import CombinedAttributesAdder, DependentImputer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestRegressor

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

    title_status_order = ['missing', 'parts only', 'salvage', 'rebuilt', 'lien', 'clean']
    condition_order = ['unknown', 'salvage', 'fair', 'good', 'excellent','like new', 'new']

    numeric_cats = ['odometer', 'cylinders']
    onehot_cats = ['transmission', 'drive', 'fuel', 'paint_color', 'type', 'manufacturer', 'state']
    ordinal_cats = ['title_status', 'condition']


    # ---------- PIPELINE ----------

    import sklearn
    sklearn.set_config(transform_output="pandas")

    prep_logic = Pipeline([
        ('features', ColumnTransformer([
            ('impute_unknown', SimpleImputer(strategy='constant', fill_value='unknown'), 
             ['condition', 'type', 'paint_color']),
            ('attr_adder', CombinedAttributesAdder(), ['year', 'odometer']),
        ], remainder='passthrough', verbose_feature_names_out=False)),

        ('scaler', ColumnTransformer([
            ('std_scaler', StandardScaler(), ['car_age', 'miles_per_year'])
        ], remainder='passthrough', verbose_feature_names_out=False))
    ])

    encoding_ct = ColumnTransformer([
        ('num', StandardScaler(), numeric_cats),
        ('ord', OrdinalEncoder(categories=[title_status_order, condition_order]), ordinal_cats),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cats)
    ], remainder='drop', verbose_feature_names_out=False) 

    full_pipeline = Pipeline([
        ('logic', prep_logic),
        ('encoding', encoding_ct),
        ('model', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    return (full_pipeline,)


@app.cell
def _(strat_test_set, strat_train_set):
    # Split train, test sets to X and Y

    X_train = strat_train_set.drop('price', axis=1)
    y_train = strat_train_set.loc[:, 'price']

    X_test = strat_train_set.drop('price', axis=1)
    y_test = strat_test_set.loc[:, 'price']
    return X_test, X_train, y_train


@app.cell
def _(X_train):
    X_train.info()
    return


@app.cell
def _(X_test, X_train, full_pipeline, y_train):
    full_pipeline.fit(X_train, y_train)

    # 3. Predict
    predictions = full_pipeline.predict(X_test)
    print("Success! Size of predictions:", len(predictions))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
