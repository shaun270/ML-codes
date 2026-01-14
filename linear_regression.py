import numpy as np

class LinearRegression:
    def __init__(self, learning_rate):
        self.W = None
        self.b = None
        self.learning_rate = learning_rate

    def fit(self, X, y, epochs):
        n = X.shape[1]
        n_samples = X.shape[0]
        print(n)
        y = y.reshape(-1,1)
        self.W = np.zeros((n, 1))
        self.b = 0.0

        for _ in range(epochs):
            y_hat = X@self.W + self.b

            dW = (2/n_samples) * (X.T @ (y_hat - y))
            db = (2/n_samples) * np.sum(y_hat - y)

            self.W -= self.learning_rate*dW
            self.b -= self.learning_rate*db
    
    def   predict(self, X):
        return X@self.W + self.b

np.random.seed(0)
X = np.random.randn(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1) * 0.5

model = LinearRegression(learning_rate=0.01)
model.fit(X, y, epochs=2000)

print(model.W, model.b)

