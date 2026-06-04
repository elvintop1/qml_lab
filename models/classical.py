from typing import Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
def run_classical_svm(X, y, test_size=0.30, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(decision_function_shape="ovo"))])
    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10, 100]},
        {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1]},
        {"svc__kernel": ["poly"], "svc__degree": [2, 3, 4], "svc__C": [0.1, 1, 10],
         "svc__gamma": ["scale", "auto", 1e-2, 1e-1], "svc__coef0": [0.0, 0.5, 1.0]}
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    yhat = best.predict(Xte)
    acc = accuracy_score(yte, yhat)
    cm = confusion_matrix(yte, yhat)
    tag = f"Classical SVM ({best.get_params()['svc__kernel']})"
    return acc, cm, tag, gs.best_params_
