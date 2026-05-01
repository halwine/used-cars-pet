from sklearn.base import BaseEstimator, TransformerMixin

class DependentImputer(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict, target_col, dependency_col):
        self.mapping_dict = mapping_dict
        self.target_col = target_col
        self.dependency_col = dependency_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        fill_values = X_copy[self.dependency_col].map(self.mapping_dict)
        X_copy[self.target_col] = X_copy[self.target_col].fillna(fill_values)
        return X_copy

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, year_ix='year', odometer_ix='odometer'):
        self.year_ix = year_ix
        self.odometer_ix = odometer_ix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Create new columns in same DataFrame
        year = X_copy[self.year_ix].astype(int)
        odometer = X_copy[self.odometer_ix].astype(float)

        X_copy["car_age"] = 2026 - year
        X_copy["miles_per_year"] = odometer / (X_copy["car_age"] + 1)
        return X_copy