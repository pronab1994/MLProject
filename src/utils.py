import os
import sys
import pickle
import tempfile

import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path) or "."
        os.makedirs(dir_name, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, dir=dir_name, suffix=".tmp") as tmp:
            pickle.dump(obj, tmp)
            temp_path = tmp.name

        os.replace(temp_path, file_path)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def _count_grid_combinations(grid):
    if not grid:
        return 1

    total = 1
    for values in grid.values():
        total *= len(values) if hasattr(values, "__len__") else 1
    return total


def evaluate_models(
    X_train, y_train, X_test, y_test,
    models, param,
    cv=3,
    random_state=42,
    grid_limit=60,
    min_iter=15,
    max_iter=60
):
    try:
        report = {}
        best_params_by_model = {}
        best_estimators = {}

        for name, model in models.items():
            logging.info(f"Tuning model: {name}")

            estimator = clone(model)
            grid = param.get(name, {}) or {}

            if not grid:
                estimator.fit(X_train, y_train)
                score = r2_score(y_test, estimator.predict(X_test))

                report[name] = score
                best_params_by_model[name] = {}
                best_estimators[name] = estimator
                continue

            combos = _count_grid_combinations(grid)

            if combos <= grid_limit:
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=grid,
                    scoring="r2",
                    cv=cv,
                    n_jobs=-1,
                    refit=True,
                )
            else:
                n_iter = int(np.clip(combos, min_iter, max_iter))
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=grid,
                    n_iter=n_iter,
                    scoring="r2",
                    cv=cv,
                    n_jobs=-1,
                    random_state=random_state,
                    refit=True,
                )

            search.fit(X_train, y_train)

            best_estimator = search.best_estimator_
            best_params = search.best_params_
            score = r2_score(y_test, best_estimator.predict(X_test))

            report[name] = score
            best_params_by_model[name] = best_params
            best_estimators[name] = best_estimator

        best_model_name = max(report, key=report.get)
        best_estimator = best_estimators[best_model_name]

        return report, best_model_name, best_estimator, best_params_by_model

    except Exception as e:
        raise CustomException(e, sys)
