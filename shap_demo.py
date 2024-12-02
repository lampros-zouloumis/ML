import lightgbm as lgb
from sklearn.model_selection import train_test_split
from feature_importance_analyzer import FeatureImportanceAnalyzer
import shap

# Load dataset
X, y = shap.datasets.adult()

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Extract feature names from the dataset
feature_names = [name.replace(" ", "_") for name in X.columns]

analyzer = FeatureImportanceAnalyzer(
    model=model,
    input_set=X_val,
    feature_names=feature_names
)

metrics = analyzer.compute_metrics()

print(metrics)

analyzer.write_metrics_to_csv()


