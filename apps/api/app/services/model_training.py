from math import ceil
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from app.schemas.models import FeatureImportance, ModelTrainingResponse


MIN_TRAINING_ROWS = 5
RANDOM_STATE = 42


class ModelTrainingError(ValueError):
    pass


def train_baseline_model(
    data: list[dict[str, Any]],
    target_column: str,
) -> ModelTrainingResponse:
    if not data:
        raise ModelTrainingError("Training data is empty")

    dataframe = pd.DataFrame(data)
    if target_column not in dataframe.columns:
        raise ModelTrainingError("Target column does not exist")

    dataframe = dataframe.dropna()
    if len(dataframe) < MIN_TRAINING_ROWS:
        raise ModelTrainingError(
            f"At least {MIN_TRAINING_ROWS} rows are required after dropping missing values"
        )

    y = dataframe[target_column]
    x = dataframe.drop(columns=[target_column])
    if x.empty:
        raise ModelTrainingError("At least one feature column is required")

    encoded_x = pd.get_dummies(x)
    if encoded_x.empty:
        raise ModelTrainingError("At least one usable feature column is required")

    test_size = max(2, ceil(len(dataframe) * 0.2))
    x_train, x_test, y_train, y_test = train_test_split(
        encoded_x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    if is_numeric_dtype(y):
        problem_type = "regression"
        model = RandomForestRegressor(random_state=RANDOM_STATE)
        metrics = _train_regression_model(model, x_train, x_test, y_train, y_test)
    else:
        problem_type = "classification"
        model = RandomForestClassifier(random_state=RANDOM_STATE)
        metrics = _train_classification_model(model, x_train, x_test, y_train, y_test)

    return ModelTrainingResponse(
        problem_type=problem_type,
        model_type=type(model).__name__,
        metrics=metrics,
        feature_importance=_build_feature_importance(
            list(encoded_x.columns),
            model.feature_importances_,
        ),
    )


def _train_regression_model(
    model: RandomForestRegressor,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "r2_score": float(r2_score(y_test, predictions)),
        "mean_absolute_error": float(mean_absolute_error(y_test, predictions)),
    }


def _train_classification_model(
    model: RandomForestClassifier,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "accuracy_score": float(accuracy_score(y_test, predictions)),
    }


def _build_feature_importance(
    feature_names: list[str],
    importances: Any,
) -> list[FeatureImportance]:
    feature_importance = [
        FeatureImportance(feature=feature, importance=float(importance))
        for feature, importance in zip(feature_names, importances, strict=True)
    ]
    return sorted(
        feature_importance,
        key=lambda item: item.importance,
        reverse=True,
    )
