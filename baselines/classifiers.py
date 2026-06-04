# qml_lab/baselines/classifiers.py
from __future__ import annotations
from typing import List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)


def _try_import_xgb(n_classes: int, random_state: int):
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None
    if n_classes > 2:
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=4,
            eval_metric="mlogloss",
            random_state=random_state,
        )
    else:
        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=4,
            eval_metric="logloss",
            random_state=random_state,
        )


def _try_import_lgbm(n_classes: int, random_state: int):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    if n_classes > 2:
        return LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=300,
            num_leaves=64,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
            random_state=random_state,
        )
    else:
        return LGBMClassifier(
            objective="binary",
            n_estimators=300,
            num_leaves=64,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
            random_state=random_state,
        )


def _try_import_cat(n_classes: int, random_state: int):
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None
    loss = "MultiClass" if n_classes > 2 else "Logloss"
    return CatBoostClassifier(
        n_estimators=300,
        depth=6,
        learning_rate=0.05,
        loss_function=loss,
        verbose=False,
        random_state=random_state,
    )


def _make_adaboost(random_state: int) -> AdaBoostClassifier:
    stump = DecisionTreeClassifier(max_depth=2, min_samples_leaf=2, random_state=random_state)
    try:
        return AdaBoostClassifier(
            estimator=stump,
            n_estimators=300,
            learning_rate=0.8,
            random_state=random_state,
        )
    except TypeError:
        return AdaBoostClassifier(
            base_estimator=stump,
            n_estimators=300,
            learning_rate=0.8,
            random_state=random_state,
        )


def build_sklearn_models(n_classes: int, random_state: int = 0, fast: bool = False) -> List[Tuple[str, object]]:
    models: List[Tuple[str, object]] = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=random_state)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("SVM Linear", SVC(kernel="linear", probability=True, random_state=random_state)),
        ("SVM Poly",   SVC(kernel="poly",   degree=3, probability=True, random_state=random_state)),
        ("SVM RBF",    SVC(kernel="rbf",    probability=True, random_state=random_state)),
        ("SVM Sigmoid",SVC(kernel="sigmoid",probability=True, random_state=random_state)),
        ("Decision Tree", DecisionTreeClassifier(random_state=random_state)),
        ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=random_state)),
        ("Extra Trees",   ExtraTreesClassifier(n_estimators=300, random_state=random_state)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=random_state)),
        ("AdaBoost", _make_adaboost(random_state)),
    ]
    xgb = _try_import_xgb(n_classes, random_state)
    if xgb is not None:
        models.append(("XGBoost", xgb))
    lgbm = _try_import_lgbm(n_classes, random_state)
    if lgbm is not None:
        models.append(("LightGBM", lgbm))
    cat = _try_import_cat(n_classes, random_state)
    if cat is not None:
        models.append(("CatBoost", cat))
    return models
