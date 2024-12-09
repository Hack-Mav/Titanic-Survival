from src.data_preprocessing import load_data, clean_data, feature_engineering
from src.model import train_logistic_regression, train_random_forest, evaluate_model
from src.visualize import plot_confusion_matrix
from sklearn.model_selection import train_test_split

# Load and clean data
data = load_data('data/raw/titanic.csv')
cleaned_data = clean_data(data)
cleaned_data = feature_engineering(cleaned_data)

# Split data into features and target
X = cleaned_data.drop('survived', axis=1)
y = cleaned_data['survived']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_logistic_regression(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
evaluate_model(y_test, y_pred)

# Visualize confusion matrix
plot_confusion_matrix(y_test, y_pred)
