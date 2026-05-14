"""
Boundary Fitter using SVR (Support Vector Regression).

Fits SVR models to boundary curve data: (xi, eta, zeta) -> (x, y, z)
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class BoundaryFitter:
    """SVR-based boundary curve fitting."""

    def __init__(self, C=1.0, epsilon=0.01, gamma='scale'):
        """
        Initialize boundary fitter.

        Args:
            C: Regularization parameter
            epsilon: Epsilon in the epsilon-SVR model
            gamma: Kernel coefficient
        """
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model_x = None
        self.model_y = None
        self.model_z = None
        self.scaler = None
        self._fitted = False

    def fit(self, xi, eta, zeta, x, y, z):
        """
        Fit SVR models for x, y, z coordinates.

        Args:
            xi, eta, zeta: Input coordinates (arrays)
            x, y, z: Target coordinates (arrays)
        """
        # Stack inputs: (xi, eta, zeta)
        X = np.column_stack([xi, eta, zeta])

        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit separate SVR for each coordinate
        self.model_x = SVR(C=self.C, epsilon=self.epsilon, gamma=self.gamma)
        self.model_x.fit(X_scaled, x)

        self.model_y = SVR(C=self.C, epsilon=self.epsilon, gamma=self.gamma)
        self.model_y.fit(X_scaled, y)

        self.model_z = SVR(C=self.C, epsilon=self.epsilon, gamma=self.gamma)
        self.model_z.fit(X_scaled, z)

        self._fitted = True

        return self

    def predict(self, xi, eta, zeta):
        """
        Predict boundary coordinates.

        Args:
            xi, eta, zeta: Scalar or array inputs

        Returns:
            Array of predicted coordinates (x, y, z)
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle scalar inputs
        is_scalar = np.isscalar(xi)
        if is_scalar:
            xi = np.array([xi])
            eta = np.array([eta])
            zeta = np.array([zeta])

        # Prepare input
        X = np.column_stack([xi, eta, zeta])
        X_scaled = self.scaler.transform(X)

        # Predict
        x_pred = self.model_x.predict(X_scaled)
        y_pred = self.model_y.predict(X_scaled)
        z_pred = self.model_z.predict(X_scaled)

        if is_scalar:
            return np.array([x_pred[0], y_pred[0], z_pred[0]])
        else:
            return np.column_stack([x_pred, y_pred, z_pred])

    def __call__(self, xi, eta, zeta):
        """Convenience method."""
        return self.predict(xi, eta, zeta)

    def __repr__(self):
        return f"BoundaryFitter(C={self.C}, epsilon={self.epsilon}, fitted={self._fitted})"