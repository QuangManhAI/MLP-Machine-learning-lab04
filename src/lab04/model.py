import numpy as np 

class MultiLayerPercepTron:
    def __init__(self, input_dim, epochs, lr):
        self.epochs = epochs
        self.lr = lr
        
        # Khoi tao trong so theo He Initialization (on dinh hon cho ReLU)
        self.W1 = np.random.randn(input_dim, 64) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, 64))

        self.W2 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros((1, 32))

        self.W3 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
        self.b3 = np.zeros((1, 1))

    def relu(self, x):
        return np.maximum(0, x)
    
    def z(self, X, b, W):
        return np.dot(X, W) + b

    def MSE(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def MAE(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    def R2(self, y_pred, y_true):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - (ss_res / (ss_tot + 1e-8))

    def MAPE(self, y_pred, y_true):
        return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0

    def fit(self, X, y):
        # Dam bao y co shape (N, 1) de khop voi phep tinh ma tran
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        self.loss_history = []
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
            self.loss_history.append(loss)

            # Backward pass
            d_loss_y_pred = 2 * (y_pred - y) / y.size
            d_loss_z3 = d_loss_y_pred
            d_loss_W3 = np.dot(a2.T, d_loss_z3)
            d_loss_b3 = np.sum(d_loss_z3, axis=0, keepdims=True)
            
            d_loss_a2 = np.dot(d_loss_z3, self.W3.T)
            d_loss_z2 = d_loss_a2 * (z2 > 0)
            d_loss_W2 = np.dot(a1.T, d_loss_z2)
            d_loss_b2 = np.sum(d_loss_z2, axis=0, keepdims=True)
            
            d_loss_a1 = np.dot(d_loss_z2, self.W2.T)
            d_loss_z1 = d_loss_a1 * (z1 > 0)
            d_loss_W1 = np.dot(X.T, d_loss_z1)
            d_loss_b1 = np.sum(d_loss_z1, axis=0, keepdims=True)

            # Gradient descent update
            self.W3 -= self.lr * d_loss_W3
            self.b3 -= self.lr * d_loss_b3
            self.W2 -= self.lr * d_loss_W2
            self.b2 -= self.lr * d_loss_b2
            self.W1 -= self.lr * d_loss_W1
            self.b1 -= self.lr * d_loss_b1          

            if epoch % 50 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {loss:.4f}")

    def predict(self, X):
        z1 = self.z(X, self.b1, self.W1)
        a1 = self.relu(z1)

        z2 = self.z(a1, self.b2, self.W2)
        a2 = self.relu(z2)

        z3 = self.z(a2, self.b3, self.W3)
        y_pred = z3
        return y_pred


class LinearRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000, fit_intercept=True):
        self.lr = lr
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Ensure y is a 2D column vector
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute and save loss
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # Compute gradients
            dw = (2.0 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2.0 / n_samples) * np.sum(y_pred - y)
            
            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class HistTreeNode:
    def __init__(self, feature=None, threshold_bin=None, threshold_val=None, left=None, right=None, value=None):
        self.feature = feature               # Index of feature to split on
        self.threshold_bin = threshold_bin   # Bin index threshold
        self.threshold_val = threshold_val   # Real threshold value
        self.left = left                     # Left child node
        self.right = right                   # Right child node
        self.value = value                   # Leaf value (mean residual prediction)
        
    def is_leaf(self):
        return self.value is not None


class HistDecisionTreeRegressorScratch:
    def __init__(self, max_depth=3, min_samples_leaf=20, max_bins=256):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.root = None

    def fit(self, X_binned, residuals, bin_thresholds):
        self.root = self._build_tree(X_binned, residuals, bin_thresholds, depth=0)
        return self

    def _build_tree(self, X_binned, residuals, bin_thresholds, depth):
        n_samples, n_features = X_binned.shape
        
        # Leaf condition: max depth reached, too few samples, or zero/near-zero variance in residuals
        if depth >= self.max_depth or n_samples < self.min_samples_leaf or np.std(residuals) < 1e-7:
            return HistTreeNode(value=np.mean(residuals))

        best_gain = -1.0
        best_feat = None
        best_bin = None
        
        # Calculate global properties for split gain
        sum_total = np.sum(residuals)
        count_total = n_samples
        
        # Search for the best split using histograms
        for feat in range(n_features):
            thresholds = bin_thresholds[feat]
            n_bins = len(thresholds) + 1
            
            # Compute histogram: sums and counts of residuals per bin using numpy's fast bincount
            bin_counts = np.bincount(X_binned[:, feat], minlength=n_bins)
            bin_sums = np.bincount(X_binned[:, feat], weights=residuals, minlength=n_bins)
            
            # Cumulative sums to evaluate all possible splits in O(n_bins)
            sum_left = np.cumsum(bin_sums)
            count_left = np.cumsum(bin_counts)
            
            sum_right = sum_total - sum_left
            count_right = count_total - count_left
            
            # Vectorized split evaluation
            # Only consider splits where both sides satisfy min_samples_leaf
            valid_split = (count_left >= self.min_samples_leaf) & (count_right >= self.min_samples_leaf)
            if not np.any(valid_split):
                continue
                
            # Avoid division by zero warnings by masking invalid splits
            with np.errstate(divide='ignore', invalid='ignore'):
                gains = (sum_left**2 / count_left) + (sum_right**2 / count_right) - (sum_total**2 / count_total)
                gains[~valid_split] = -1.0
                
            max_gain_idx = np.argmax(gains)
            if gains[max_gain_idx] > best_gain:
                best_gain = gains[max_gain_idx]
                best_feat = feat
                best_bin = max_gain_idx
                
        # If no valid split improves loss, return leaf
        if best_gain <= 0.0 or best_feat is None:
            return HistTreeNode(value=np.mean(residuals))
            
        # Perform split
        left_mask = X_binned[:, best_feat] <= best_bin
        right_mask = ~left_mask
        
        # Translate the best binned split threshold back to a real-value threshold
        threshold_val = bin_thresholds[best_feat][best_bin] if best_bin < len(bin_thresholds[best_feat]) else bin_thresholds[best_feat][-1]
        
        left_child = self._build_tree(X_binned[left_mask], residuals[left_mask], bin_thresholds, depth + 1)
        right_child = self._build_tree(X_binned[right_mask], residuals[right_mask], bin_thresholds, depth + 1)
        
        return HistTreeNode(feature=best_feat, threshold_bin=best_bin, threshold_val=threshold_val, left=left_child, right=right_child)

    def predict(self, X):
        return self._predict_batch(self.root, X)

    def _predict_batch(self, node, X):
        if node.is_leaf():
            return np.full(X.shape[0], node.value)
            
        preds = np.zeros(X.shape[0])
        left_mask = X[:, node.feature] <= node.threshold_val
        right_mask = ~left_mask
        
        if np.any(left_mask):
            preds[left_mask] = self._predict_batch(node.left, X[left_mask])
        if np.any(right_mask):
            preds[right_mask] = self._predict_batch(node.right, X[right_mask])
            
        return preds


class HistGradientBoostingRegressorScratch:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=20, max_bins=256):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.trees = []
        self.base_pred = None
        self.bin_thresholds = []

    def fit(self, X, y):
        # Handle target shape
        if len(y.shape) == 1:
            y = y.astype(float)
        else:
            y = y.ravel().astype(float)
            
        X = X.astype(float)
        n_samples, n_features = X.shape

        # Step 1: Pre-compute bin thresholds and map training X to integer bin indices
        self.bin_thresholds = []
        X_binned = np.zeros_like(X, dtype=int)
        for j in range(n_features):
            unique_vals = np.unique(X[:, j])
            if len(unique_vals) <= self.max_bins:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            else:
                percentiles = np.linspace(0, 100, self.max_bins + 1)[1:-1]
                thresholds = np.percentile(X[:, j], percentiles)
                thresholds = np.unique(thresholds)
            
            if len(thresholds) == 0:
                thresholds = np.array([unique_vals[0]])
                
            self.bin_thresholds.append(thresholds)
            X_binned[:, j] = np.digitize(X[:, j], thresholds)

        # Step 2: Initialize base prediction (mean of target values)
        self.base_pred = np.mean(y)
        y_pred = np.full(n_samples, self.base_pred)
        
        self.loss_history = []
        self.loss_history.append(np.mean((y - y_pred) ** 2))

        self.trees = []
        # Step 3: Sequential boosting iterations
        for m in range(self.n_estimators):
            # Compute negative gradient of MSE loss (residuals)
            residuals = y - y_pred
            
            # Fit tree to residuals using the pre-binned features
            tree = HistDecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_bins=self.max_bins
            )
            tree.fit(X_binned, residuals, self.bin_thresholds)
            
            # Update prediction values for the next iteration
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            
            # Save loss after update
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        X = X.astype(float)
        preds = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return preds


class ModelChecker:
    @staticmethod
    def run_all_checks():
        print("="*70)
        print("RUNNING BENCHMARK COMPARISONS (FROM SCRATCH vs. STANDARD LIBRARIES)")
        print("="*70)
        
        # Import necessary verification tools
        import time
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            torch_available = True
        except ImportError:
            torch_available = False
            print("[Warning] PyTorch is not available. Skipping PyTorch MLP comparison.")
            
        def r2_score_scratch(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1.0 - (ss_res / (ss_tot + 1e-8))

        def mape_scratch(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0

        # 1. Generate regression dataset
        X, y = make_regression(n_samples=1500, n_features=8, noise=0.1, random_state=42)
        y = y.reshape(-1, 1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale input/output data
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        
        # -------------------------------------------------------------
        # 1. Linear Regression Comparison
        # -------------------------------------------------------------
        print("\n--- 1. Linear Regression Comparison ---")
        from sklearn.linear_model import LinearRegression as SklearnLinearRegression
        
        # Sklearn
        start = time.time()
        lr_sk = SklearnLinearRegression()
        lr_sk.fit(X_train_scaled, y_train_scaled)
        time_sk = time.time() - start
        y_pred_sk = lr_sk.predict(X_test_scaled)
        mse_sk = np.mean((y_pred_sk - y_test_scaled) ** 2)
        r2_sk = r2_score_scratch(y_test_scaled, y_pred_sk)
        mape_sk = mape_scratch(y_test_scaled, y_pred_sk)
        
        # Scratch
        start = time.time()
        lr_scratch = LinearRegressionScratch(lr=0.05, epochs=1000)
        lr_scratch.fit(X_train_scaled, y_train_scaled)
        time_scratch = time.time() - start
        y_pred_scratch = lr_scratch.predict(X_test_scaled)
        mse_scratch = np.mean((y_pred_scratch - y_test_scaled) ** 2)
        r2_scratch = r2_score_scratch(y_test_scaled, y_pred_scratch)
        mape_scratch_val = mape_scratch(y_test_scaled, y_pred_scratch)
        
        print(f"sklearn Linear Regression - Test MSE: {mse_sk:.6f} | R2: {r2_sk:.6f} | MAPE: {mape_sk:.4f}% | Time: {time_sk:.4f}s")
        print(f"Scratch Linear Regression - Test MSE: {mse_scratch:.6f} | R2: {r2_scratch:.6f} | MAPE: {mape_scratch_val:.4f}% | Time: {time_scratch:.4f}s")
        
        # -------------------------------------------------------------
        # 2. HistGradientBoosting Comparison
        # -------------------------------------------------------------
        print("\n--- 2. HistGradientBoosting Comparison ---")
        from sklearn.ensemble import HistGradientBoostingRegressor as SklearnHistGB
        
        # Sklearn
        start = time.time()
        hgb_sk = SklearnHistGB(max_iter=50, max_depth=3, min_samples_leaf=20, random_state=42)
        hgb_sk.fit(X_train_scaled, y_train_scaled.ravel())
        time_sk = time.time() - start
        y_pred_sk = hgb_sk.predict(X_test_scaled).reshape(-1, 1)
        mse_sk = np.mean((y_pred_sk - y_test_scaled) ** 2)
        r2_sk = r2_score_scratch(y_test_scaled, y_pred_sk)
        mape_sk = mape_scratch(y_test_scaled, y_pred_sk)
        
        # Scratch
        start = time.time()
        hgb_scratch = HistGradientBoostingRegressorScratch(n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=20)
        hgb_scratch.fit(X_train_scaled, y_train_scaled.ravel())
        time_scratch = time.time() - start
        y_pred_scratch = hgb_scratch.predict(X_test_scaled).reshape(-1, 1)
        mse_scratch = np.mean((y_pred_scratch - y_test_scaled) ** 2)
        r2_scratch = r2_score_scratch(y_test_scaled, y_pred_scratch)
        mape_scratch_val = mape_scratch(y_test_scaled, y_pred_scratch)
        
        print(f"sklearn HistGradientBoosting - Test MSE: {mse_sk:.6f} | R2: {r2_sk:.6f} | MAPE: {mape_sk:.4f}% | Time: {time_sk:.4f}s")
        print(f"Scratch HistGradientBoosting - Test MSE: {mse_scratch:.6f} | R2: {r2_scratch:.6f} | MAPE: {mape_scratch_val:.4f}% | Time: {time_scratch:.4f}s")

        # -------------------------------------------------------------
        # 3. MLP Comparison (Custom vs PyTorch)
        # -------------------------------------------------------------
        if torch_available:
            print("\n--- 3. MLP (Multi-Layer Perceptron) Comparison ---")
            epochs_mlp = 2000
            lr_mlp = 0.05
            
            # Define PyTorch MLP Module matching our scratch design
            class PyTorchMLP(nn.Module):
                def __init__(self, input_dim):
                    super(PyTorchMLP, self).__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                    )
                    # Use He (Kaiming) normal initialization
                    for m in self.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0.0)

                def forward(self, x):
                    return self.net(x)

            # PyTorch MLP training
            start = time.time()
            X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
            
            torch_model = PyTorchMLP(X_train_scaled.shape[1])
            optimizer = optim.SGD(torch_model.parameters(), lr=lr_mlp)
            criterion = nn.MSELoss()
            
            for epoch in range(epochs_mlp):
                optimizer.zero_grad()
                outputs = torch_model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
                
            time_torch = time.time() - start
            
            # Test PyTorch MLP
            with torch.no_grad():
                X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
                y_pred_torch = torch_model(X_test_t).numpy()
            mse_torch = np.mean((y_pred_torch - y_test_scaled) ** 2)
            r2_torch = r2_score_scratch(y_test_scaled, y_pred_torch)
            mape_torch = mape_scratch(y_test_scaled, y_pred_torch)
            
            # Scratch MLP training (suppressing progress print to keep logs clean)
            import sys
            import os
            
            # Save original stdout to suppress printing during custom fit
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                start = time.time()
                mlp_scratch = MultiLayerPercepTron(input_dim=X_train_scaled.shape[1], epochs=epochs_mlp, lr=lr_mlp)
                mlp_scratch.fit(X_train_scaled, y_train_scaled)
                time_scratch = time.time() - start
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
                
            # Test Scratch MLP
            y_pred_scratch = mlp_scratch.predict(X_test_scaled)
            mse_scratch = np.mean((y_pred_scratch - y_test_scaled) ** 2)
            r2_scratch = r2_score_scratch(y_test_scaled, y_pred_scratch)
            mape_scratch_val = mape_scratch(y_test_scaled, y_pred_scratch)
            
            print(f"PyTorch MLP - Test MSE: {mse_torch:.6f} | R2: {r2_torch:.6f} | MAPE: {mape_torch:.4f}% | Time: {time_torch:.4f}s")
            print(f"Scratch MLP - Test MSE: {mse_scratch:.6f} | R2: {r2_scratch:.6f} | MAPE: {mape_scratch_val:.4f}% | Time: {time_scratch:.4f}s")
            
        print("="*70)

