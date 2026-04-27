from sklearn.base import BaseEstimator, TransformerMixin

class DependentImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in a target column based on values from a dependency column
    using a provided mapping dictionary.
    """

    def __init__(self, mapping_dict, target_col, dependency_col):
        self.mapping_dict = mapping_dict
        self.target_col = target_col
        self.dependency_col = dependency_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Generate fill values by mapping the dependency column to the dictionary
        fill_values = X_copy[self.dependency_col].map(self.mapping_dict)
        # Fill NaNs only in the target column where they exist
        X_copy[self.target_col] = X_copy[self.target_col].fillna(fill_values)
        return X_copy


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Calculates new features: 'car_age' and 'miles_per_year'.
    Drops the original 'year' column after calculation.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Calculate vehicle age based on the target year (2026)
        X_copy['car_age'] = 2026 - X_copy['year']
        # Calculate average mileage per year, adding 1 to avoid division by zero
        X_copy['miles_per_year'] = X_copy['odometer'] / (X_copy['car_age'] + 1)
        # Drop 'year' as it is now redundant
        return X_copy.drop('year', axis=1)