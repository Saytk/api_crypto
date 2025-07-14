"""
Custom implementation of metrics and utilities to replace sklearn dependencies.

This module provides custom implementations of:
1. Classification metrics (accuracy, precision, recall, f1_score)
2. Log loss and ROC/AUC metrics
3. Confusion matrix
4. Class weight calculation
5. Cross-validation utilities
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true: Ground truth (correct) target values
        y_pred: Estimated targets as returned by a classifier

    Returns:
        Confusion matrix in the form [[TN, FP], [FN, TP]]
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Get unique classes
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)

    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Fill confusion matrix
    for i in range(len(y_true)):
        cm[int(y_true[i]), int(y_pred[i])] += 1

    return cm


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy classification score.

    Args:
        y_true: Ground truth (correct) target values
        y_pred: Estimated targets as returned by a classifier

    Returns:
        Accuracy of the predictions
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """
    Calculate precision score.

    Args:
        y_true: Ground truth (correct) target values
        y_pred: Estimated targets as returned by a classifier
        zero_division: Value to return when there is a zero division

    Returns:
        Precision of the positive class
    """
    cm = confusion_matrix(y_true, y_pred)

    if len(cm) <= 1:
        return 1.0

    tp = cm[1, 1]
    fp = cm[0, 1]

    if tp + fp == 0:
        return zero_division

    return tp / (tp + fp)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """
    Calculate recall score.

    Args:
        y_true: Ground truth (correct) target values
        y_pred: Estimated targets as returned by a classifier
        zero_division: Value to return when there is a zero division

    Returns:
        Recall of the positive class
    """
    cm = confusion_matrix(y_true, y_pred)

    if len(cm) <= 1:
        return 1.0

    tp = cm[1, 1]
    fn = cm[1, 0]

    if tp + fn == 0:
        return zero_division

    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """
    Calculate F1 score.

    Args:
        y_true: Ground truth (correct) target values
        y_pred: Estimated targets as returned by a classifier
        zero_division: Value to return when there is a zero division

    Returns:
        F1 score of the positive class
    """
    precision = precision_score(y_true, y_pred, zero_division)
    recall = recall_score(y_true, y_pred, zero_division)

    if precision + recall == 0:
        return zero_division

    return 2 * (precision * recall) / (precision + recall)


def log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    Calculate log loss, aka logistic loss or cross-entropy loss.

    Args:
        y_true: Ground truth (correct) target values
        y_pred: Predicted probabilities
        eps: Small constant to avoid log(0)

    Returns:
        Log loss averaged over samples
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clip probabilities to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss
    losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return np.mean(losses)


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true: Ground truth (correct) target values
        y_score: Target scores (probability estimates of the positive class)

    Returns:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Thresholds used to compute fpr and tpr
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Get sorted scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Get unique thresholds
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    thresholds = y_score[threshold_idxs]

    # Initialize tpr and fpr
    tpr = np.zeros(threshold_idxs.size + 1)
    fpr = np.zeros(threshold_idxs.size + 1)

    # Accumulate true positives and false positives
    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # Calculate rates
    tpr[:-1] = tps / n_pos if n_pos > 0 else 0
    fpr[:-1] = fps / n_neg if n_neg > 0 else 0

    # Add (1, 1) point
    tpr[-1] = 1.0
    fpr[-1] = 1.0

    return fpr, tpr, thresholds


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Area Under the Curve (AUC) using the trapezoidal rule.

    Args:
        x: x coordinates (e.g., false positive rates)
        y: y coordinates (e.g., true positive rates)

    Returns:
        Area under the curve
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Sort values by x
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Compute AUC using the trapezoidal rule
    area = np.trapz(y, x)

    return area


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true: Ground truth (correct) target values
        y_score: Target scores (probability estimates of the positive class)

    Returns:
        Area under the ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute average precision (AP) from prediction scores.

    Args:
        y_true: Ground truth (correct) target values
        y_score: Target scores (probability estimates of the positive class)

    Returns:
        Average precision score
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Return average precision
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Accumulate true positives
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return 0

    # Compute precision and recall at each threshold
    precision = np.zeros(threshold_idxs.size + 1)
    recall = np.zeros(threshold_idxs.size + 1)

    # Accumulate true positives
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    precision[:-1] = tps / (tps + fps)
    recall[:-1] = tps / n_pos

    # Add (1, 0) point
    precision[-1] = 1.0
    recall[-1] = 0.0

    # Compute average precision as the weighted mean of precisions
    ap = np.sum(np.diff(recall) * precision[:-1])

    return ap


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, output_dict: bool = False) -> Union[str, Dict]:
    """
    Build a text report showing the main classification metrics.

    Args:
        y_true: Ground truth (correct) target values
        y_pred: Estimated targets as returned by a classifier
        output_dict: If True, return output as dict

    Returns:
        Text summary of the precision, recall, F1 score for each class,
        or dictionary if output_dict=True
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Get unique classes
    classes = np.unique(np.concatenate((y_true, y_pred)))

    # Calculate metrics for each class
    metrics = {}
    for cls in classes:
        y_true_cls = (y_true == cls).astype(int)
        y_pred_cls = (y_pred == cls).astype(int)

        precision = precision_score(y_true_cls, y_pred_cls)
        recall = recall_score(y_true_cls, y_pred_cls)
        f1 = f1_score(y_true_cls, y_pred_cls)
        support = np.sum(y_true_cls)

        metrics[str(int(cls))] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Create output
    if output_dict:
        result = metrics.copy()
        result["accuracy"] = accuracy
        return result

    # Format text report
    report = "              precision    recall  f1-score   support\n\n"

    for cls in sorted(metrics.keys()):
        m = metrics[cls]
        report += f"{cls:>14s}       {m['precision']:.2f}      {m['recall']:.2f}      {m['f1-score']:.2f}      {m['support']}\n"

    report += f"\n    accuracy                           {accuracy:.2f}      {len(y_true)}\n"

    return report


def compute_class_weight(class_weight: str, classes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute class weights for unbalanced datasets.

    Args:
        class_weight: 'balanced' to use balanced weights
        classes: Array of unique class labels
        y: Array of class labels

    Returns:
        Array of class weights
    """
    if class_weight != 'balanced':
        return np.ones(len(classes))

    # Count samples in each class
    y = np.asarray(y)
    class_counts = np.bincount(y.astype(int))

    # Compute weights
    n_samples = len(y)
    n_classes = len(classes)

    weights = n_samples / (n_classes * class_counts)

    return weights


class KFold:
    """
    K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        """
        Initialize K-Folds cross-validator.

        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle the data before splitting
            random_state: Random seed for reproducibility when shuffle=True
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.

        Args:
            X: Training data
            y: Target variable (ignored)

        Returns:
            List of tuples (train_idx, test_idx) where train_idx and test_idx are arrays of indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Shuffle if requested
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)

        # Create folds
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        splits = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            splits.append((train_indices, test_indices))
            current = stop

        return splits


def cross_val_score(estimator: object, X: np.ndarray, y: np.ndarray, cv: Union[int, object] = 5) -> np.ndarray:
    """
    Evaluate a score by cross-validation.

    Args:
        estimator: Estimator object implementing 'fit' and 'predict'
        X: Training data
        y: Target variable
        cv: Cross-validation strategy (int for KFold or object with split method)

    Returns:
        Array of scores of the estimator for each fold
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Create CV splitter if cv is an integer
    if isinstance(cv, int):
        cv = KFold(n_splits=cv)

    # Get splits
    splits = cv.split(X, y)

    # Evaluate each fold
    scores = []
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit and predict
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        # Calculate accuracy
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

    return np.array(scores)


class LogisticRegression:
    """
    Simple implementation of Logistic Regression classifier.
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize Logistic Regression classifier.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Apply sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, n_iterations: int = 1000) -> 'LogisticRegression':
        """
        Fit the model according to the given training data.

        Args:
            X: Training data
            y: Target values
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations for gradient descent

        Returns:
            self
        """
        # Initialize parameters
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Probability estimates for samples in X.

        Args:
            X: Samples

        Returns:
            Array of probabilities for both classes [negative, positive]
        """
        linear_model = np.dot(X, self.weights) + self.bias
        pos_probs = self._sigmoid(linear_model)
        # Create a 2D array with probabilities for both classes [negative, positive]
        neg_probs = 1 - pos_probs
        return np.column_stack((neg_probs, pos_probs))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Samples

        Returns:
            Predicted class labels
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)
