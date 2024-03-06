import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular

# Generate a synthetic dataset for demonstration
np.random.seed(42)
X = np.random.rand(100, 5)
y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(int)  # Binary classification

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Choose an instance for explanation
instance_idx = 10
test_instance = X[instance_idx].reshape(1, -1)

# Define feature names and class names for LIME
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
class_names = ['Class 0', 'Class 1']

# Create LIME explainer and generate explanations
explainer = lime_tabular.LimeTabularExplainer(X, mode="classification", feature_names=feature_names, class_names=class_names)
explanation = explainer.explain_instance(test_instance.flatten(), model.predict_proba, num_features=5)

# Visualize the explanation
explanation.show_in_notebook()
