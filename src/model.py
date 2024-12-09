from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Logistic Regression
def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Random Forest
def train_random_forest(X_train, y_train):
    """Train a random forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Support Vector Machine (SVM)
def train_svm(X_train, y_train):
    """Train a Support Vector Machine classifier."""
    model = SVC(kernel='linear', random_state=42)  # You can experiment with different kernels like 'rbf'
    model.fit(X_train, y_train)
    return model

# K-Nearest Neighbors (KNN)
def train_knn(X_train, y_train):
    """Train a K-Nearest Neighbors classifier."""
    model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
    model.fit(X_train, y_train)
    return model

# Gradient Boosting
def train_gradient_boosting(X_train, y_train):
    """Train a Gradient Boosting classifier."""
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# XGBoost
def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier."""
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluation Function
def evaluate_model(y_test, y_pred):
    """Evaluate the model's performance."""
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
