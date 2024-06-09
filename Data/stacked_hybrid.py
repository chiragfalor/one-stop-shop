import numpy as np
import pandas as pd
from typing import List, Any

from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class StackedHybridRegressor:
    """
    A stacked hybrid model that uses multiple base models to generate features for a meta model.
    """
    def __init__(self, base_models: List[RegressorMixin], meta_model: RegressorMixin):
        """
        Initialize the StackedHybrid model with a list of base models and a meta model.

        Args:
        base_models (List[RegressorMixin]): List of regression models.
        meta_model (RegressorMixin): A single regression model used as the final estimator.
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.y_columns = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model using the provided training data.

        Args:
        X (pd.DataFrame): Training features.
        y (pd.DataFrame): Target variable.
        """
        y = y.squeeze()  # Ensure y is 1D for scikit-learn compatibility
        self.y_columns = [y.name] if isinstance(y, pd.Series) else y.columns if isinstance(y, pd.DataFrame) else ['target']
        for model in self.base_models:
            model.fit(X, y)

        X_meta = np.column_stack([model.predict(X) for model in self.base_models])
        self.meta_model.fit(X_meta, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the stacked model.

        Args:
        X (pd.DataFrame): Features for prediction.

        Returns:
        pd.DataFrame: Predicted values.
        """
        X_meta = np.column_stack([model.predict(X) for model in self.base_models])
        y_pred = pd.DataFrame(self.meta_model.predict(X_meta), columns=self.y_columns, index=X.index)
        return y_pred

class BoostedRegressor:
    """
    A hybrid model that sequentially fits models and reduces the residual errors in predictions.
    """
    def __init__(self, models: List[RegressorMixin]):
        """
        Initialize the BoostedHybrid model with a list of models.

        Args:
        models (List[RegressorMixin]): List of regression models.
        """
        self.models = models
        self.y_columns = None

    def fit(self, Xs: List[pd.DataFrame], y: pd.DataFrame) -> None:
        """
        Fit the model using the provided training data for each model.

        Args:
        Xs (List[pd.DataFrame]): List of training features for each model.
        y (pd.DataFrame): Target variable.
        """
        assert len(Xs) == len(self.models), "Number of feature sets must match the number of models."
        y = y.squeeze()  # Ensure y is 1D for scikit-learn compatibility
        self.y_columns = [y.name] if isinstance(y, pd.Series) else y.columns if isinstance(y, pd.DataFrame) else ['target']

        y_res = y.copy()
        for model, X in zip(self.models, Xs):
            model.fit(X, y_res)
            y_res -= model.predict(X)

    def predict(self, Xs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Predict using the boosted model.

        Args:
        Xs (List[pd.DataFrame]): List of features for prediction for each model.

        Returns:
        pd.DataFrame: Predicted values.
        """
        assert len(Xs) == len(self.models), "Number of feature sets must match the number of models."
        y_pred = np.zeros(Xs[0].shape[0])
        for model, X in zip(self.models, Xs):
            y_pred += model.predict(X)
        y_pred = pd.DataFrame(y_pred, columns=self.y_columns, index=Xs[0].index)
        return y_pred
    
class MajorityClassifier:
    """
    A hybrid model that uses majority voting to predict the class labels.
    """
    def __init__(self, models: List[ClassifierMixin]):
        """
        Initialize the MajorityClassifier model with a list of models.

        Args:
        models (List[ClassifierMixin]): List of classification models.
        """
        self.models = models
        self.y_columns = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model using the provided training data for each model.

        Args:
        X (pd.DataFrame): Training features.
        y (pd.DataFrame): Target variable.
        """
        y = y.squeeze()
        self.y_columns = [y.name] if isinstance(y, pd.Series) else y.columns if isinstance(y, pd.DataFrame) else ['target']
        for model in self.models:
            model.fit(X, y)


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the majority classifier model.

        Args:
        X (pd.DataFrame): Features for prediction.

        Returns:
        pd.DataFrame: Predicted class labels.
        """
        y_pred = self._raw_predict(X)
        y_pred = y_pred.mode(axis=1)[0].astype(int)
        y_pred.name = self.y_columns[0]
        return y_pred
    
    def _raw_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using the majority classifier model.

        Args:
        X (pd.DataFrame): Features for prediction.

        Returns:
        pd.DataFrame: Predicted class labels.
        """
        y_pred = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            y_pred[:, i] = model.predict(X)
        y_pred = pd.DataFrame(y_pred, index = X.index, columns = [f"{model.__class__.__name__}_{i}" for i, model in enumerate(self.models)], dtype=int)
        return y_pred
    
    def prediction_accuracy(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Calculate the prediction accuracy of the model.

        Args:
        X (pd.DataFrame): Features for prediction.
        y (pd.DataFrame): Actual class labels.

        Returns:
        float: Prediction accuracy.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    

# Example usage
if __name__ == "__main__":
    # Load sample data
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=["resp"])

    # Initialize stacked hybrid model
    base_models = [
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        ExtraTreesRegressor(),
        XGBRegressor()
    ]
    meta_model = LinearRegression()
    stacked_model = StackedHybridRegressor(base_models, meta_model)

    # Fit and predict
    stacked_model.fit(X, y)
    y_pred = stacked_model.predict(X)
    # print actual and predicted values
    print(y.head())
    print(y_pred.head())

    # Initialize boosted hybrid model
    models = [
        LinearRegression(),
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        XGBRegressor()
    ]
    boosted_model = BoostedRegressor(models)

    # Fit and predict
    Xs = [X] * len(models)
    boosted_model.fit(Xs, y)
    y_pred = boosted_model.predict(Xs)
    # print actual and predicted values
    print(y.head())
    print(y_pred.head())

    # classification dataset
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=["target"])

    # Initialize majority classifier model
    models = [
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        ExtraTreesClassifier(),
        LogisticRegression(),
        XGBClassifier(),

    ]
    majority_model = MajorityClassifier(models)

    # Fit and predict
    majority_model.fit(X, y)
    y_pred = majority_model.predict(X)
    # print actual and predicted values
    print(y.head())
    print(y_pred.head())
