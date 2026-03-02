import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np

    return (pd,)


@app.cell
def _():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("austinreese/craigslist-carstrucks-data")

    print("Path to dataset files:", path)
    return (path,)


@app.cell
def _(path, pd):
    df = pd.read_csv(path + "/" + "vehicles.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
