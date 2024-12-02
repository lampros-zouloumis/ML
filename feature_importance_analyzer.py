import numpy as np
import shap
import csv

class FeatureImportanceAnalyzer:
    """
        Globally important features: Features with a high absolute mean

        Stable features (low std): Consistently influential across the dataset.
        Volatile features (high std): Feature impact changes significantly across samples, possibly due to interactions or dataset subgroups.
    """
    def __init__(self, model, input_set, feature_names):
        self.model = model
        self.input_set = input_set
        self.feature_names = feature_names

        
    def compute_metrics(self):
        """
        Compute absolute mean and standard deviation for SHAP values.

        :return: Dictionary with feature-wise metrics.
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values= explainer.shap_values(self.input_set)
        mean_magnitude = np.mean(np.abs(shap_values), axis=0)
        std_dev = np.std(shap_values, axis=0)
        self.metrics = {
            "mean_magnitude": mean_magnitude,
            "std_dev": std_dev
        }
        return self.metrics

    def analyze_features(self, threshold=0.05):
        """
        Analyze features based on absolute mean and standard deviation.

        :param threshold: Threshold for identifying important features.
        :return: Features exceeding the threshold for absolute mean.
        """

        self.validate_feature_names()
        important_features = []
        for i, feature in enumerate(self.feature_names):
            if self.metrics["mean_magnitude"][i] > threshold:
                important_features.append({
                    "feature": feature,
                    "mean_magnitude": self.metrics["mean_magnitude"][i],
                    "std_dev": self.metrics["std_dev"][i],
                })
        return important_features
    
    def validate_feature_names(self):
        """
        Validate the feature names.

        :param feature_names: List of feature names.
        :param input_set: Dataset to validate against.
        :return: Validated feature names.
        """
        if len(self.feature_names) != len(self.input_set[0]):
            raise ValueError("Number of feature names must match the number of features in the input dataset.")
        return self.feature_names
    

    def write_metrics_to_csv(self, file_name="shap_metrics.csv"):
        """
        Write the computed metrics (mean magnitude and standard deviation) to a CSV file.

        :param file_name: Name of the output CSV file (default: 'shap_metrics.csv').
        """
        if not hasattr(self, "metrics"):
            raise ValueError("Metrics have not been computed. Call compute_metrics() first.")
        
        data = [
            {
                "feature": self.feature_names[i],
                "mean_magnitude": self.metrics["mean_magnitude"][i],
                "std_dev": self.metrics["std_dev"][i],
            }
            for i in range(len(self.feature_names))
        ]

        with open(file_name, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["feature", "mean_magnitude", "std_dev"])
            writer.writeheader()
            writer.writerows(data)

        print(f"Metrics have been written to '{file_name}'.")