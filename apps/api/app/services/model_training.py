from math import ceil
import json
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Dataset, ModelRun
from app.schemas.models import FeatureImportance, ModelRunRead, ModelTrainingResponse


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
        task_type=problem_type,
        model_type=type(model).__name__,
        target_column=target_column,
        metrics=metrics,
        feature_importance=_build_feature_importance(
            list(encoded_x.columns),
            model.feature_importances_,
        ),
    )


def train_and_persist_baseline_model(
    db: Session,
    project_id: int,
    dataset: Dataset,
    target_column: str,
) -> ModelTrainingResponse:
    rows = json.loads(dataset.raw_data_json)
    result = train_baseline_model(rows, target_column)
    model_run = ModelRun(
        project_id=project_id,
        dataset_id=dataset.id,
        target_column=target_column,
        task_type=result.problem_type,
        model_type=result.model_type,
        metrics_json=json.dumps(result.metrics),
        feature_importance_json=json.dumps(
            [item.model_dump() for item in result.feature_importance]
        ),
    )
    db.add(model_run)
    db.commit()
    db.refresh(model_run)

    return ModelTrainingResponse(
        model_run_id=model_run.id,
        project_id=project_id,
        dataset_id=dataset.id,
        problem_type=result.problem_type,
        task_type=result.problem_type,
        model_type=result.model_type,
        target_column=target_column,
        metrics=result.metrics,
        feature_importance=result.feature_importance,
    )


def list_project_model_runs(db: Session, project_id: int) -> list[ModelRun]:
    result = db.execute(
        select(ModelRun)
        .where(ModelRun.project_id == project_id)
        .order_by(ModelRun.created_at.desc(), ModelRun.id.desc())
    )
    return list(result.scalars().all())


def model_run_to_read_data(model_run: ModelRun) -> ModelRunRead:
    return ModelRunRead(
        id=model_run.id,
        project_id=model_run.project_id,
        dataset_id=model_run.dataset_id,
        target_column=model_run.target_column,
        task_type=model_run.task_type,
        model_type=model_run.model_type,
        metrics=json.loads(model_run.metrics_json),
        feature_importance=[
            FeatureImportance(**item)
            for item in json.loads(model_run.feature_importance_json)
        ],
        created_at=model_run.created_at,
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
