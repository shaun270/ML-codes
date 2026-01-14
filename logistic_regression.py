import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        self.w = np.zeros((n_features, 1))
        self.b = 0.0

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            y_hat = self.sigmoid(z)

            error = y_hat - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

np.random.seed(0)

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# train
model = LogisticRegression(learning_rate=0.1, epochs=2000)
model.fit(X, y)

X_new = np.array([[6, 7], [7, 8]])
y_pred = model.predict(X_new)

print(y_pred)

