import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Set random seed
np.random.seed(42)

# Configuration
N_SAMPLES = 1000
OUTPUT_DIR = "data"
DATA_FILE = os.path.join(OUTPUT_DIR, "chip_defect_data.csv")
METRICS_FILE = os.path.join(OUTPUT_DIR, "random_forest_metrics.txt")

def generate_synthetic_data():
    """Generate synthetic chip design data."""
    data = {
        'chip_id': [f'CHIP_{i}' for i in range(N_SAMPLES)],
        'transistor_count': np.random.randint(1_000_000, 10_000_000, N_SAMPLES),
        'defect_rate': np.random.uniform(0.0, 0.1, N_SAMPLES),
        'power_efficiency': np.random.uniform(0.5, 2.0, N_SAMPLES),
        'has_defect': np.random.choice([0, 1], N_SAMPLES, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    return df

def train_random_forest(df):
    """Train Random Forest classifier."""
    X = df[['transistor_count', 'defect_rate', 'power_efficiency']]
    y = df['has_defect']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    with open(METRICS_FILE, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
    
    return accuracy

def main():
    """Generate data and train model."""
    df = generate_synthetic_data()
    accuracy = train_random_forest(df)
    print(f"Dataset saved to {DATA_FILE}")
    print(f"Model metrics saved to {METRICS_FILE}")
    print(f"Random Forest Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()