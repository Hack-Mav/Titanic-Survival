from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import numpy as np

def tune_random_forest(X_train, y_train):
    """Tune RandomForest model with GridSearchCV"""
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy')
    grid_search_rf.fit(X_train, y_train)

    print("Best Parameters (Random Forest):", grid_search_rf.best_params_)
    print("Best Score (Random Forest):", grid_search_rf.best_score_)

    return grid_search_rf.best_estimator_

def tune_svm(X_train, y_train):
    """Tune SVM model with GridSearchCV"""
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
    }

    svm = SVC(random_state=42)
    grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy')
    grid_search_svm.fit(X_train, y_train)

    print("Best Parameters (SVM):", grid_search_svm.best_params_)
    print("Best Score (SVM):", grid_search_svm.best_score_)

    return grid_search_svm.best_estimator_

def tune_knn(X_train, y_train):
    """Tune KNN model with GridSearchCV"""
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 10],
        'metric': ['euclidean', 'manhattan'],
    }

    knn = KNeighborsClassifier()
    grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, scoring='accuracy')
    grid_search_knn.fit(X_train, y_train)

    print("Best Parameters (KNN):", grid_search_knn.best_params_)
    print("Best Score (KNN):", grid_search_knn.best_score_)

    return grid_search_knn.best_estimator_

def tune_xgboost(X_train, y_train):
    """Tune XGBoost model with RandomizedSearchCV"""
    param_dist_xgb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 10],
        'subsample': [0.7, 0.8, 1.0],
    }

    xgb = XGBClassifier(random_state=42)
    random_search_xgb = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist_xgb, n_iter=10, cv=5, scoring='accuracy', random_state=42)
    random_search_xgb.fit(X_train, y_train)

    print("Best Parameters (XGBoost):", random_search_xgb.best_params_)
    print("Best Score (XGBoost):", random_search_xgb.best_score_)

    return random_search_xgb.best_estimator_

def cross_validate_model(model, X, y):
    """Evaluate a model with cross-validation"""
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores):.2f}")
    return cv_scores
