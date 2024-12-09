import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load raw Titanic dataset from the specified file path."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the Titanic dataset by handling missing values and encoding categorical variables."""
    # Fill missing age values with the median
    df['age'].fillna(df['age'].median(), inplace=True)
    
    # Drop 'name', 'ticket', 'cabin', and 'embarked' columns (or any irrelevant columns)
    df.drop(['embark_town', 'class', 'embarked', 'who', 'deck', 'alive'], axis=1, inplace=True)
    
    # Convert 'sex' into a binary variable (0 = male, 1 = female)
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})

    # Convert 'alone' into a binary variable (0 = False, 1 = True)
    # df['alone'] = df['alone'].map({'False': 0, 'True': 1})
    
    # Optionally: Scale numerical features
    features = ['age', 'sibsp', 'parch', 'fare']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Save the cleaned data
    df.to_csv('data/processed/train_cleaned.csv', index=False)
    
    return df

def feature_engineering(df):
    """Create new features from existing ones (optional)."""
    df['family_size'] = df['sibsp'] + df['parch']
    df['is_alone'] = (df['family_size'] == 0).astype(int)
    return df
