"""
Conformal Prediction for Uncertainty Calibration
=================================================

Implements conformal prediction methods to ensure valid coverage:
1. Split Conformal: Calibrate on held-out calibration set
2. Rolling Conformal: Adaptive calibration using recent residuals

Guarantees: If data is exchangeable, coverage ≥ (1-α) with high probability

Reference: 
- Vovk et al. (2005) "Algorithmic Learning in a Random World"
- Shafer & Vovk (2008) "A Tutorial on Conformal Prediction"
"""

import numpy as np
from typing import Tuple, Optional


class ConformalPredictor:
    """Base class for conformal prediction"""
    
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.calibration_scores = []
    
    def calibrate(self, residuals):
        """
        Calibrate on residuals from calibration set
        
        Args:
            residuals: Array of absolute residuals |y_true - y_pred|
        """
        raise NotImplementedError
    
    def get_interval(self, point_prediction):
        """
        Get prediction interval around point prediction
        
        Args:
            point_prediction: Point forecast value
            
        Returns:
            (lower_bound, upper_bound)
        """
        raise NotImplementedError


class SplitConformalPredictor(ConformalPredictor):
    """
    Split Conformal Prediction
    
    Steps:
    1. Split data: Train | Calibration | Test
    2. Fit model on Train data
    3. Compute residuals on Calibration data
    4. Use quantile of calibration residuals for test intervals
    """
    
    def __init__(self, alpha=0.1, multi_alpha=None):
        """
        Args:
            alpha: Primary miscoverage level
            multi_alpha: List of additional alpha levels (e.g., [0.05, 0.2] for 95% and 80%)
        """
        super().__init__(alpha)
        self.quantile = None
        self.multi_alpha = multi_alpha or []
        self.multi_quantiles = {}  # Store quantiles for each alpha
    
    def calibrate(self, residuals):
        """
        Compute conformity score (quantile of absolute residuals)
        
        Args:
            residuals: Calibration residuals |y_cal - pred_cal|
        """
        if len(residuals) == 0:
            raise ValueError("No calibration residuals provided")
        
        self.calibration_scores = np.abs(residuals)
        
        # Compute (1-alpha) quantile with finite-sample correction
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)  # Cap at 1.0
        
        self.quantile = np.quantile(self.calibration_scores, q_level)
        
        # Compute quantiles for additional alphas
        for alpha in self.multi_alpha:
            q_level_i = np.ceil((n + 1) * (1 - alpha)) / n
            q_level_i = min(q_level_i, 1.0)
            self.multi_quantiles[alpha] = np.quantile(self.calibration_scores, q_level_i)
        
        return self
    
    def get_interval(self, point_prediction):
        """Return symmetric interval around point prediction"""
        if self.quantile is None:
            raise ValueError("Model not calibrated. Call calibrate() first.")
        
        lower = point_prediction - self.quantile
        upper = point_prediction + self.quantile
        
        return lower, upper
    
    def get_batch_intervals(self, point_predictions):
        """Get intervals for multiple predictions"""
        if self.quantile is None:
            raise ValueError("Model not calibrated")
        
        predictions = np.array(point_predictions)
        lower = predictions - self.quantile
        upper = predictions + self.quantile
        
        return lower, upper
    
    def get_multi_intervals(self, point_prediction):
        """
        Get prediction intervals for all alpha levels
        
        Returns:
            dict: {alpha: (lower, upper)}
        """
        if self.quantile is None:
            raise ValueError("Model not calibrated")
        
        intervals = {}
        
        # Primary alpha
        intervals[self.alpha] = (
            point_prediction - self.quantile,
            point_prediction + self.quantile
        )
        
        # Additional alphas
        for alpha, quantile in self.multi_quantiles.items():
            intervals[alpha] = (
                point_prediction - quantile,
                point_prediction + quantile
            )
        
        return intervals


class RollingConformalPredictor(ConformalPredictor):
    """
    Rolling Conformal Prediction (Adaptive)
    
    Maintains a rolling window of recent residuals for calibration.
    Updates quantile as new residuals become available.
    
    Better for non-stationary data where error distribution changes over time.
    """
    
    def __init__(self, alpha=0.1, window_size=50):
        """
        Args:
            alpha: Miscoverage level
            window_size: Number of recent residuals to keep
        """
        super().__init__(alpha)
        self.window_size = window_size
        self.residual_window = []
    
    def update(self, new_residual):
        """
        Update rolling window with new residual
        
        Args:
            new_residual: Latest absolute residual
        """
        self.residual_window.append(abs(new_residual))
        
        # Keep only last window_size residuals
        if len(self.residual_window) > self.window_size:
            self.residual_window.pop(0)
    
    def calibrate(self, residuals):
        """Initialize window with calibration residuals"""
        self.residual_window = list(np.abs(residuals))
        
        # Keep only last window_size
        if len(self.residual_window) > self.window_size:
            self.residual_window = self.residual_window[-self.window_size:]
        
        return self
    
    def get_interval(self, point_prediction):
        """Compute interval using current window"""
        if len(self.residual_window) == 0:
            raise ValueError("No calibration data. Call calibrate() or update() first.")
        
        # Compute quantile from current window
        n = len(self.residual_window)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        quantile = np.quantile(self.residual_window, q_level)
        
        lower = point_prediction - quantile
        upper = point_prediction + quantile
        
        return lower, upper
    
    def get_current_quantile(self):
        """Get current conformity quantile"""
        if len(self.residual_window) == 0:
            return None
        
        n = len(self.residual_window)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        return np.quantile(self.residual_window, q_level)


class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive Conformal Inference (ACI)
    
    Adjusts coverage dynamically based on observed miscoverage.
    Reference: Gibbs & Candes (2021) "Adaptive Conformal Inference"
    """
    
    def __init__(self, alpha=0.1, gamma=0.005, window_size=50):
        """
        Args:
            alpha: Target miscoverage level
            gamma: Learning rate for quantile adjustment
            window_size: Rolling window size for residuals
        """
        super().__init__(alpha)
        self.gamma = gamma
        self.window_size = window_size
        self.residual_window = []
        self.alpha_t = alpha  # Adaptive alpha
    
    def update(self, new_residual, y_true, interval):
        """
        Update with new observation and coverage information
        
        Args:
            new_residual: |y_true - y_pred|
            y_true: True value
            interval: (lower, upper) prediction interval
        """
        self.residual_window.append(abs(new_residual))
        
        if len(self.residual_window) > self.window_size:
            self.residual_window.pop(0)
        
        # Check if y_true was covered
        covered = (interval[0] <= y_true <= interval[1])
        
        # Update alpha_t based on coverage
        if covered:
            # Decrease alpha_t (make intervals tighter)
            self.alpha_t = self.alpha_t + self.gamma * (self.alpha - 0)
        else:
            # Increase alpha_t (make intervals wider)
            self.alpha_t = self.alpha_t + self.gamma * (self.alpha - 1)
        
        # Clip to valid range
        self.alpha_t = np.clip(self.alpha_t, 0.001, 0.5)
    
    def calibrate(self, residuals):
        """Initialize with calibration residuals"""
        self.residual_window = list(np.abs(residuals))
        
        if len(self.residual_window) > self.window_size:
            self.residual_window = self.residual_window[-self.window_size:]
        
        return self
    
    def get_interval(self, point_prediction):
        """Compute interval using adaptive alpha"""
        if len(self.residual_window) == 0:
            raise ValueError("No calibration data")
        
        n = len(self.residual_window)
        q_level = np.ceil((n + 1) * (1 - self.alpha_t)) / n
        q_level = min(q_level, 1.0)
        
        quantile = np.quantile(self.residual_window, q_level)
        
        lower = point_prediction - quantile
        upper = point_prediction + quantile
        
        return lower, upper


class BootstrapConformalPredictor:
    """
    Combine bootstrap sampling with conformal calibration
    
    Steps:
    1. Generate bootstrap predictions
    2. Use conformal calibration to adjust intervals
    """
    
    def __init__(self, alpha=0.1, n_bootstrap=100):
        """
        Args:
            alpha: Miscoverage level
            n_bootstrap: Number of bootstrap samples
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.conformal = SplitConformalPredictor(alpha)
    
    def calibrate(self, residuals):
        """Calibrate conformal predictor"""
        self.conformal.calibrate(residuals)
        return self
    
    def get_interval_from_bootstrap(self, bootstrap_predictions):
        """
        Get conformal-calibrated interval from bootstrap samples
        
        Args:
            bootstrap_predictions: Array of bootstrap prediction samples
            
        Returns:
            (lower, upper, mean_prediction)
        """
        # Use bootstrap mean as point prediction
        point_pred = np.mean(bootstrap_predictions)
        
        # Get conformal interval
        lower, upper = self.conformal.get_interval(point_pred)
        
        return lower, upper, point_pred


def compute_coverage_metrics(y_true, intervals_lower, intervals_upper):
    """
    Compute coverage metrics for prediction intervals
    
    Args:
        y_true: True values (n,)
        intervals_lower: Lower bounds (n,)
        intervals_upper: Upper bounds (n,)
        
    Returns:
        Dictionary with coverage metrics
    """
    y_true = np.array(y_true)
    intervals_lower = np.array(intervals_lower)
    intervals_upper = np.array(intervals_upper)
    
    # Check coverage
    covered = (intervals_lower <= y_true) & (y_true <= intervals_upper)
    coverage_rate = np.mean(covered)
    
    # Interval widths
    widths = intervals_upper - intervals_lower
    mean_width = np.mean(widths)
    
    # Winkler score (coverage-adjusted interval width)
    alpha = 0.1  # Assuming 90% nominal coverage
    winkler_scores = []
    for i in range(len(y_true)):
        width = widths[i]
        if y_true[i] < intervals_lower[i]:
            penalty = 2 / alpha * (intervals_lower[i] - y_true[i])
        elif y_true[i] > intervals_upper[i]:
            penalty = 2 / alpha * (y_true[i] - intervals_upper[i])
        else:
            penalty = 0
        winkler_scores.append(width + penalty)
    
    mean_winkler = np.mean(winkler_scores)
    
    return {
        'coverage_rate': coverage_rate,
        'mean_interval_width': mean_width,
        'mean_winkler_score': mean_winkler,
        'num_samples': len(y_true)
    }


if __name__ == "__main__":
    # Quick test
    print("Testing conformal predictors...")
    
    # Simulate data
    np.random.seed(42)
    n_cal = 50
    n_test = 20
    
    # Calibration: residuals from validation set
    cal_residuals = np.abs(np.random.randn(n_cal))
    
    # Test predictions
    test_predictions = np.random.randn(n_test) * 2
    test_true = test_predictions + np.random.randn(n_test)
    
    # Test Split Conformal
    print("\n1. Split Conformal (90% coverage):")
    split_cp = SplitConformalPredictor(alpha=0.1)
    split_cp.calibrate(cal_residuals)
    
    lower, upper = split_cp.get_batch_intervals(test_predictions)
    metrics = compute_coverage_metrics(test_true, lower, upper)
    print(f"   Coverage: {metrics['coverage_rate']:.2%} (target: 90%)")
    print(f"   Mean width: {metrics['mean_interval_width']:.3f}")
    
    # Test Rolling Conformal
    print("\n2. Rolling Conformal (90% coverage):")
    rolling_cp = RollingConformalPredictor(alpha=0.1, window_size=30)
    rolling_cp.calibrate(cal_residuals)
    
    covered_count = 0
    for i in range(n_test):
        lower_i, upper_i = rolling_cp.get_interval(test_predictions[i])
        if lower_i <= test_true[i] <= upper_i:
            covered_count += 1
        # Update with new residual
        rolling_cp.update(abs(test_true[i] - test_predictions[i]))
    
    print(f"   Coverage: {covered_count / n_test:.2%} (target: 90%)")
    
    # Test Adaptive Conformal
    print("\n3. Adaptive Conformal (90% coverage):")
    adaptive_cp = AdaptiveConformalPredictor(alpha=0.1, gamma=0.01, window_size=30)
    adaptive_cp.calibrate(cal_residuals)
    
    covered_count = 0
    for i in range(n_test):
        lower_i, upper_i = adaptive_cp.get_interval(test_predictions[i])
        if lower_i <= test_true[i] <= upper_i:
            covered_count += 1
        # Update with coverage information
        adaptive_cp.update(
            abs(test_true[i] - test_predictions[i]),
            test_true[i],
            (lower_i, upper_i)
        )
    
    print(f"   Coverage: {covered_count / n_test:.2%} (target: 90%)")
    print(f"   Final alpha_t: {adaptive_cp.alpha_t:.4f}")
    
    print("\nConformal prediction test complete!")
