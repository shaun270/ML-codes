import numpy as np

class SVM:
    def __init__(self, lr, iterations, C, lambda_param):
        self.lr = lr
        self.iterations = iterations
        self.C = C
        self.lambda_param = lambda_param

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y == 0, -1, 1).astype(float)

        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        for _ in range(self.iterations):
            for idx, x_i in enumerate(X):
                y_i = np.dot(x_i, self.w) + self.b

                if max(0, 1 - y[idx] * y_i) == 0:
                    self.w -= self.lr * 2 * self.lambda_param * self.w
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y[idx] * x_i)
                    self.b += self.lr * y[idx]

    def predict(self, X):
        out = np.dot(X, self.w) + self.b
        return np.where(out >= 0, 1, -1)
        

import numpy as np

np.random.seed(0)

# two separable clusters
N = 100
X_pos = np.random.randn(N, 2) + np.array([2, 2])
X_neg = np.random.randn(N, 2) + np.array([-2, -2])

X = np.vstack((X_pos, X_neg))
y = np.hstack((np.ones(N), np.zeros(N)))  # 1 and 0 labels

svm = SVM(
    lr=0.001,
    iterations=1000,
    C=1.0,
    lambda_param=0.01
)

svm.fit(X, y)

preds = svm.predict(X)

# convert labels to same format
y_true = np.where(y == 0, -1, 1)

accuracy = np.mean(preds == y_true)
print("Accuracy:", accuracy)
print("Weights:", svm.w)
print("Bias:", svm.b)
