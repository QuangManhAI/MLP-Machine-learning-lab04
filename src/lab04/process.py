from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Indices of numerical features in the pipeline input array (after imputer)
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No training required
        
    def transform(self, X):
        # Calculate engineered features
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        
        # Concat new features with the input matrix
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
