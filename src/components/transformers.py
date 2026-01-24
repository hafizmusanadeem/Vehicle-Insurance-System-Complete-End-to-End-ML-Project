import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GenderMapper(BaseEstimator, TransformerMixin):
    """Maps Gender column: Female → 0, Male → 1"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1})
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops columns based on schema"""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")