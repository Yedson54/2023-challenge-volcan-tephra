import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

def missings_type_col(X: pd.DataFrame):
    majors_col_idx = list(range(8))
    traces_col_idx = list(range(9, 35))
    X_ = X.copy()

    if isinstance(X_, np.ndarray):
        X_ = pd.DataFrame(X_)
    msno_majors = X_.iloc[:, majors_col_idx].isna().sum(axis=1)
    msno_traces = X_.iloc[:, traces_col_idx].isna().sum(axis=1)
    X_.insert(len(X_.columns), "msno_", msno_majors > msno_traces)

    return X_.values

def group_features(X: pd.DataFrame, feature_groups: dict):
    """
    Note:
        Should be placed after `missing_type_col` (recommended) but before
        StandardScaler() (mandatory to ensure Scaling is applied on the new features)
    """
    X_ = X.copy()
    if isinstance(X_, np.ndarray):
        X_ = pd.DataFrame(X_)
    for name, group in feature_groups.items():
        group_df = X_[[g for g in group if g in X_]]

        if not group_df.empty:
            X_[f"min({name})"] = group_df.apply(np.min, axis=1)
            X_[f"max({name})"] = group_df.apply(np.max, axis=1)
            X_[f"mean({name})"] = group_df.apply(np.mean, axis=1)
            X_[f"std({name})"] = group_df.apply(np.std, axis=1)
            X_[f"median({name})"] = group_df.apply(np.median, axis=1)
            X_[f"mode({name})"] = stats.mode(group_df, axis=1, keepdims=True)[0]

    return X_.values

class Classifier(BaseEstimator):
    def __init__(self):
        feature_groups = {"majors": list(range(8)),
                          "traces": list(range(9, 35))}

        msno_col_transformer = FunctionTransformer(missings_type_col)
        group_features_transformer = FunctionTransformer(
            group_features, kw_args={"feature_groups": feature_groups}
        )

        self.transformer = Pipeline(steps=[
            ("msno_col", msno_col_transformer),
            # ("imputer", SimpleImputer(strategy="mean")),
            # ("imputer", KNNImputer(weights='distance')),
            # ("group_features", group_features_transformer),
            # ("scaler", StandardScaler()),
            # ("None", None)
        ])
        self.model = HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42, warm_start=True, l2_regularization=1
        )
        
        self.pipe = make_pipeline(self.transformer, self.model)

    def fit(self, X, y):
        self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)
