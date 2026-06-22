import numpy as np

class MultiLayerPercepTron:
    def __init__(self, epochs, lr):
        self.epochs = epochs
        self.lr = lr
        self.W1 = np.random.rand((10, 64))
        self.b1 = np.random.rand(10)

        self.W2 = np.random.rand((64, 32))
        self.b2 = np.random.rand(32)

        self.W3 = np.random.rand((32, 1))
        self.b3 = np.random.rand(1)
    def relu(self, x):
        return np.maximum(0, x)
    
    def z(self, X, b, W):
        return np.dot(X, W) + b
    def MSE(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    def fit(self, X, y):
        for epoch in range(self.epochs):
            # Forward pass
            z1 = self.z(X, self.b1, self.W1)
            a1 = self.relu(z1)

            z2 = self.z(a1, self.b2, self.W2)
            a2 = self.relu(z2)

            z3 = self.z(a2, self.b3, self.W3)
            y_pred = z3

            # Compute loss
            loss = self.MSE(y_pred, y)

            # Backward pass (simplified)
            d_loss_y_pred = 2 * (y_pred - y) / y.size
            d_loss_z3 = d_loss_y_pred
            d_loss_W3 = np.dot(a2.T, d_loss_z3)
            d_loss_b3 = np.sum(d_loss_z3, axis=0)
            d_loss_a2 = np.dot(d_loss_z3, self.W3.T)
            d_loss_z2 = d_loss_a2 * (z2 > 0)
            d_loss_W2 = np.dot(a1.T, d_loss_z2)
            d_loss_b2 = np.sum(d_loss_z2, axis=0)
            d_loss_a1 = np.dot(d_loss_z2, self.W2.T)
            d_loss_z1 = d_loss_a1 * (z1 > 0)
            d_loss_W1 = np.dot(X.T, d_loss_z1)
            d_loss_b1 = np.sum(d_loss_z1, axis=0)

            # gradient descent update
            self.W3 -= self.lr * d_loss_W3
            self.b3 -= self.lr * d_loss_b3
            self.W2 -= self.lr * d_loss_W2
            self.b2 -= self.lr * d_loss_b2
            self.W1 -= self.lr * d_loss_W1
            self.b1 -= self.lr * d_loss_b1          

    def predict(self, X):
        z1 = self.z(X, self.b1, self.W1)
        a1 = self.relu(z1)

        z2 = self.z(a1, self.b2, self.W2)
        a2 = self.relu(z2)

        z3 = self.z(a2, self.b3, self.W3)
        y_pred = z3
        return y_pred