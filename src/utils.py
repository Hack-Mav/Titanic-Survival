import joblib

def save_model(model, filename):
    """Save the trained model using joblib."""
    joblib.dump(model, filename)

def load_model(filename):
    """Load a saved model using joblib."""
    return joblib.load(filename)
