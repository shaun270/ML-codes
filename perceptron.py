import numpy as np

class MLP:
    def __init__(self, layer_sizes, lr=0.1, seed=0):
        """
        layer_sizes: list like [D, H1, H2, ..., K]
          - Regression: K = 1
          - Binary:     K = 1 (logit)
          - Multiclass: K = C (logits)
        """
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.sizes = layer_sizes

        self.W = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # small init (helps stability)
            Wi = rng.normal(0, 0.1, size=(fan_in, fan_out))
            bi = np.zeros((1, fan_out))
            self.W.append(Wi)
            self.b.append(bi)

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _tanh_grad(a):
        # a = tanh(z) already computed
        return 1.0 - a**2

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _softmax(logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, X):
        """
        Returns output logits and caches for backprop
        """
        A = X
        caches = {"A": [A], "Z": []}

        # hidden layers: tanh
        for i in range(len(self.W) - 1):
            Z = A @ self.W[i] + self.b[i]
            A = self._tanh(Z)
            caches["Z"].append(Z)
            caches["A"].append(A)

        # last layer: linear logits (no activation here)
        ZL = A @ self.W[-1] + self.b[-1]
        caches["Z"].append(ZL)
        return ZL, caches

    def _backward(self, X, dZL, caches):
        """
        dZL: gradient w.r.t last-layer logits, shape (N, K)
        """
        N = X.shape[0]
        dW = [None] * len(self.W)
        db = [None] * len(self.b)

        # last layer grads
        A_prev = caches["A"][-1]  # last hidden activation
        dW[-1] = (A_prev.T @ dZL) / N
        db[-1] = np.sum(dZL, axis=0, keepdims=True) / N

        # propagate to previous layer
        dA = dZL @ self.W[-1].T

        # hidden layers backwards
        for i in reversed(range(len(self.W) - 1)):
            A_i = caches["A"][i + 1]  # activation at layer i (after tanh)
            dZ = dA * self._tanh_grad(A_i)

            A_prev = caches["A"][i]   # activation before this layer
            dW[i] = (A_prev.T @ dZ) / N
            db[i] = np.sum(dZ, axis=0, keepdims=True) / N

            dA = dZ @ self.W[i].T

        return dW, db

    def _step(self, dW, db):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * dW[i]
            self.b[i] -= self.lr * db[i]

    # ---------- Losses + fit modes ----------

    def fit_regression(self, X, y, epochs=1000):
        y = y.reshape(-1, 1)
        for _ in range(epochs):
            pred, caches = self.forward(X)  # pred shape (N,1)
            # MSE loss gradient: d(pred) = 2*(pred-y)
            dZL = 2.0 * (pred - y)
            dW, db = self._backward(X, dZL, caches)
            self._step(dW, db)

    def fit_binary(self, X, y, epochs=1000):
        """
        Binary classification with BCE-with-logits
        Output layer has K=1 (logit)
        """
        y = y.reshape(-1, 1)
        for _ in range(epochs):
            logits, caches = self.forward(X)        # (N,1)
            probs = self._sigmoid(logits)           # (N,1)

            # BCE-with-logits gradient simplifies to (probs - y)
            dZL = (probs - y)
            dW, db = self._backward(X, dZL, caches)
            self._step(dW, db)

    def fit_multiclass(self, X, y, epochs=1000):
        """
        Multiclass with softmax cross-entropy
        Output layer has K=C logits
        y: integer class labels (N,)
        """
        y = y.reshape(-1)
        N = X.shape[0]
        C = self.sizes[-1]

        for _ in range(epochs):
            logits, caches = self.forward(X)        # (N,C)
            probs = self._softmax(logits)           # (N,C)

            # one-hot
            Y = np.zeros((N, C))
            Y[np.arange(N), y] = 1.0

            # softmax+CE gradient: (probs - Y)
            dZL = (probs - Y)
            dW, db = self._backward(X, dZL, caches)
            self._step(dW, db)

    # ---------- Predict helpers ----------

    def predict_regression(self, X):
        pred, _ = self.forward(X)
        return pred

    def predict_binary(self, X, threshold=0.5):
        logits, _ = self.forward(X)
        probs = self._sigmoid(logits)
        return (probs >= threshold).astype(int)

    def predict_multiclass(self, X):
        logits, _ = self.forward(X)
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)

np.random.seed(0)
X = np.random.randn(200, 1)
y = 3*X[:,0] + 2 + 0.5*np.random.randn(200)

net = MLP([1, 8, 1], lr=0.05)
net.fit_regression(X, y, epochs=2000)

pred = net.predict_regression(X).reshape(-1)
print("MSE:", np.mean((pred - y)**2))
