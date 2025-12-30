import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.params = None
        self.cost_history = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        # Add a bias term to the features
        X = np.insert(X, 0, 1, axis=1).astype(float)

        # Initialize parameters
        self.params = np.zeros(X.shape[1])
        self.cost_history = np.zeros(self.iterations)
        m = len(y)

        for i in range(self.iterations):
            self.params = self.params - (self.learning_rate/m) * (X.T @ (self._sigmoid(X @ self.params) - y))
            self.cost_history[i] = self._cost_function(self._sigmoid(X @ self.params), y)

    def predict(self, X, threshold=0.5):
        # Add a bias term to the features
        X = np.insert(X, 0, 1, axis=1).astype(float)
        return (self._sigmoid(X @ self.params) >= threshold).astype(int)

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.weights) + self.bias)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        true_positives = np.sum((predictions == 1) & (y == 1))
        false_positives = np.sum((predictions == 1) & (y == 0))
        false_negatives = np.sum((predictions == 0) & (y == 1))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }