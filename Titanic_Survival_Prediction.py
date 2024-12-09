"""from src.data_preprocessing import load_data, clean_data, feature_engineering
from src.model import (train_logistic_regression, train_random_forest, 
                       train_svm, train_knn, train_gradient_boosting, 
                       train_xgboost, evaluate_model)
from sklearn.model_selection import train_test_split

# Load and clean data
data = load_data('data/raw/train.csv')
cleaned_data = clean_data(data)
cleaned_data = feature_engineering(cleaned_data)

# Split data into features and target
X = cleaned_data.drop('survived', axis=1)
y = cleaned_data['survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models and evaluate
models = {
    "Logistic Regression": train_logistic_regression(X_train, y_train),
    "Random Forest": train_random_forest(X_train, y_train),
    "SVM": train_svm(X_train, y_train),
    "K-Nearest Neighbors": train_knn(X_train, y_train),
    "Gradient Boosting": train_gradient_boosting(X_train, y_train),
    "XGBoost": train_xgboost(X_train, y_train)
}

# Evaluate each model
for model_name, model in models.items():
    print(f"\nEvaluating {model_name}:")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)
"""


from src.data_preprocessing import load_data, clean_data, feature_engineering
from src.model import train_logistic_regression, train_random_forest, train_svm, train_knn, train_gradient_boosting, train_xgboost, evaluate_model
from src.tuning import tune_random_forest, tune_svm, tune_knn, tune_xgboost, cross_validate_model
from sklearn.model_selection import train_test_split

# Load and clean data
data = load_data('data/raw/titanic.csv')
cleaned_data = clean_data(data)
cleaned_data = feature_engineering(cleaned_data)

# Split data into features and target
X = cleaned_data.drop('survived', axis=1)
y = cleaned_data['survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tuning models
print("Tuning Random Forest...")
best_rf_model = tune_random_forest(X_train, y_train)

print("\nTuning SVM...")
best_svm_model = tune_svm(X_train, y_train)

print("\nTuning KNN...")
best_knn_model = tune_knn(X_train, y_train)

# print("\nTuning XGBoost...")
# best_xgb_model = tune_xgboost(X_train, y_train)

# Perform cross-validation to check performance of tuned models
print("\nCross-validation for Best Random Forest Model...")
cross_validate_model(best_rf_model, X, y)

print("\nCross-validation for Best SVM Model...")
cross_validate_model(best_svm_model, X, y)

print("\nCross-validation for Best KNN Model...")
cross_validate_model(best_knn_model, X, y)

# print("\nCross-validation for Best XGBoost Model...")
# cross_validate_model(best_xgb_model, X, y)

# Evaluate on test data with the best model (e.g., Random Forest)
y_pred_rf = best_rf_model.predict(X_test)
print("\nEvaluating Best Random Forest Model on Test Data...")
evaluate_model(y_test, y_pred_rf)
