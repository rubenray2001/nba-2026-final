from sklearn.ensemble import VotingRegressor
import xgboost as xgb
from sklearn.base import is_regressor
import numpy as np

# Mock data
X = np.random.rand(10, 5)
y = np.random.rand(10)

# Init model
xgb_model = xgb.XGBRegressor(objective='reg:absoluteerror', n_jobs=1)

print(f"Is regressor? {is_regressor(xgb_model)}")

try:
    ensemble = VotingRegressor(estimators=[('xgb', xgb_model)])
    ensemble.fit(X, y)
    print("VotingRegressor: SUCCESS")
except Exception as e:
    print(f"VotingRegressor: FAILED - {e}")

# Try wrapping
from sklearn.base import BaseEstimator, RegressorMixin

class XGBWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def get_params(self, deep=True):
        return self.model.get_params(deep)
        
    def set_params(self, **params):
        self.model.set_params(**params)
        return self
        
    # Crucial: Define _estimator_type correctly
    @property
    def _estimator_type(self):
        return "regressor"

print("\nTesting Wrapper...")
wrapper = XGBWrapper(objective='reg:absoluteerror', n_jobs=1)
print(f"Wrapper is regressor? {is_regressor(wrapper)}")

try:
    ensemble = VotingRegressor(estimators=[('xgb_wrap', wrapper)])
    ensemble.fit(X, y)
    print("VotingRegressor (Wrapper): SUCCESS")
except Exception as e:
    print(f"VotingRegressor (Wrapper): FAILED - {e}")
