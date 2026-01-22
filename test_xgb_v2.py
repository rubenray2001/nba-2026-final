from sklearn.base import BaseEstimator, RegressorMixin, is_regressor
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import numpy as np

# Wrapper attempt 2
class SafeXGBRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor" 
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None
        
    def fit(self, X, y, **kwargs):
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y, **kwargs)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def get_params(self, deep=True):
        return self.params
        
    def set_params(self, **params):
        self.params.update(params)
        return self
    
    # New Sklearn tags
    def __sklearn_tags__(self):
        from sklearn.utils._tags import RegressorTags
        return RegressorTags()

wrapper = SafeXGBRegressor(n_jobs=1)
print(f"Wrapper v2 is regressor? {is_regressor(wrapper)}")

# Try standard
estimator = xgb.XGBRegressor()
estimator._estimator_type = "regressor"
print(f"Patched XGB is regressor? {is_regressor(estimator)}")
