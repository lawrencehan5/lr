import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse


class LogisticRegression:
    def __init__(self, target_type="binary", C=1.0, max_iter=1000, tol=1e-7,
                 device="cpu", threshold=0.5, history_size=10, max_eval=None,
                 dtype=torch.float64, tolerance_change=None):
        if target_type not in ("binary", "continuous", "categorical"):
            raise ValueError("Invalid target_type")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use float32 or float64")
        if not (C > 0 and math.isfinite(tol) and tol > 0 and 0 < threshold < 1):
            raise ValueError("Invalid C, tol or threshold")
        if min(max_iter, history_size) < 1 or (max_eval is not None and max_eval < 1):
            raise ValueError("Iteration limits and history_size must be positive")
        self.target_type = target_type
        self.C = float(C)
        self.dtype = dtype
        self.device = torch.device(device)
        self.threshold = threshold
        self.tol = tol
        if tolerance_change is None:
            tolerance_change = 1e-9 if dtype == torch.float32 else 1e-15
        if not math.isfinite(tolerance_change) or tolerance_change <= 0:
            raise ValueError("Invalid tolerance_change")
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Use an explicit CPU or CUDA device")
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.history_size = history_size
        self.tolerance_change = tolerance_change
        self._fitted = False

    @staticmethod
    def _check_deterministic():
        if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
            raise RuntimeError("Enable torch.use_deterministic_algorithms(True, warn_only=False)")

    def _to_tensor(self, values, dtype=None):
        return torch.as_tensor(values, dtype=dtype, device=self.device)

    @staticmethod
    def _check_numeric(values):
        if values.dtype.kind not in "buif" or (values.size and not (
            np.isfinite(values.min()) and np.isfinite(values.max())
        )):
            raise ValueError("Expected finite real numeric values")

    def _to_csr(self, matrix):
        index_dtype = torch.int32 if max(*matrix.shape, matrix.nnz) <= 2**31 - 1 else torch.int64
        return torch.sparse_csr_tensor(
            self._to_tensor(matrix.indptr, index_dtype), self._to_tensor(matrix.indices, index_dtype),
            self._to_tensor(matrix.data, self.dtype), size=matrix.shape, device=self.device)

    def _prepare_features(self, *, X_dense=None, X_binary=None,
                          X_sparse=None, training=False):
        X_dense, X_binary = (None if block is None else np.asarray(block)
                             for block in (X_dense, X_binary))
        if X_sparse is not None and (not sparse.issparse(X_sparse) or X_sparse.format != "csr"):
            raise ValueError("X_sparse must be SciPy CSR")
        inputs = (X_dense, X_binary, X_sparse)
        present = [block for block in inputs if block is not None]
        if not present or any(block.ndim != 2 for block in present):
            raise ValueError("Provide two-dimensional feature matrices")
        n_samples = present[0].shape[0]
        if (training and not n_samples) or any(block.shape[0] != n_samples for block in present):
            raise ValueError("Invalid or inconsistent row counts")
        feature_schema = tuple(0 if block is None else block.shape[1] for block in inputs)
        if not training and feature_schema != self._feature_schema:
            raise ValueError("Features must match the fitted feature schema")

        feature_blocks, feature_start = [], 0
        for feature_type, features, count in zip(("dense", "binary", "sparse"), inputs, feature_schema):
            if not count:
                continue
            transpose = None
            if feature_type == "dense":
                features = self._to_tensor(features, self.dtype)
            elif feature_type == "binary":
                if features.dtype != np.bool_:
                    raise ValueError("X_binary must have Boolean dtype")
                features = self._to_tensor(features.T, torch.bool).contiguous()
            else:
                if not features.has_canonical_format:
                    raise ValueError("Use canonical SciPy CSR")
                transpose = self._to_csr(features.T.tocsr()) if training else None
                features = self._to_csr(features)
            feature_blocks.append((feature_type, features, transpose, slice(feature_start, feature_start + count)))
            feature_start += count
        if not feature_start:
            raise ValueError("At least one feature is required")
        return feature_blocks, n_samples, feature_schema, feature_start

    def _vector(self, values, n_samples):
        values = np.asarray(values)
        if values.shape == (n_samples, 1):
            values = values[:, 0]
        if values.shape != (n_samples,):
            raise ValueError("Expected one value per observation")
        self._check_numeric(values)
        return values

    def _encode_target(self, y, n_samples):
        y = np.asarray(y)
        if y.shape != (n_samples,):
            raise ValueError("y must be one-dimensional with one value per observation")
        self._check_numeric(y)
        if self.target_type == "continuous":
            if y.min() < 0 or y.max() > 1:
                raise ValueError("Fractional target must be in [0, 1]")
            self.classes_ = None
            return self._to_tensor(y, self.dtype), 1
        if self.target_type == "binary":
            negative, positive = y.min(), y.max()
            if negative == positive or (y.dtype != np.bool_ and not np.all((y == negative) | (y == positive))):
                raise ValueError("Expected two target classes")
            self.classes_ = np.array([negative, positive], dtype=y.dtype)
            return self._to_tensor(y if y.dtype == np.bool_ else y == positive, torch.bool), 1
        self.classes_ = np.unique(y)
        if len(self.classes_) < 3:
            raise ValueError("Expected at least three target classes")
        return self._to_tensor(np.searchsorted(self.classes_, y), torch.int64), len(self.classes_)

    def _logits(self, feature_blocks, n_samples):
        logits = None
        for feature_type, features, _, coef_slice in feature_blocks:
            weights = self.W_[coef_slice]
            if feature_type == "dense":
                logits = torch.addmm(self.b_, features, weights)
                continue
            if logits is None:
                logits = self.b_.expand(n_samples, -1).clone()
            if feature_type == "sparse":
                logits.add_(torch.sparse.mm(features, weights))
            else:
                for indicator, weight in zip(features, weights):
                    logits.add_(torch.where(indicator[:, None], weight, 0.0))
        return logits

    def _loss_and_residual(self, logits, y, sample_weight):
        if self.target_type == "categorical":
            residual = F.log_softmax(logits, dim=1)
            class_indices = y[:, None]
            losses = residual.gather(1, class_indices)[:, 0].neg_()
            loss = losses.sum() if sample_weight is None else torch.dot(losses, sample_weight)
            del losses
            residual.exp_()
            residual.scatter_(1, class_indices, residual.gather(1, class_indices).sub_(1))
        elif self.target_type == "binary":
            logits = logits[:, 0]
            torch.where(y, -logits, logits, out=logits)
            losses = F.softplus(logits, threshold=40 if self.dtype == torch.float64 else 20)
            loss = losses.sum() if sample_weight is None else torch.dot(losses, sample_weight)
            del losses
            logits.sigmoid_()
            torch.where(y, -logits, logits, out=logits)
            residual = logits[:, None]
        else:
            logits = logits[:, 0]
            loss = F.binary_cross_entropy_with_logits(
                logits, y, weight=sample_weight, reduction="sum"
            )
            residual = logits.sigmoid_().sub_(y)[:, None]

        if sample_weight is not None:
            residual.mul_(sample_weight[:, None])
        return loss, residual

    @torch.no_grad()
    def fit(self, y, *, X_dense=None, X_binary=None, X_sparse=None,
            sample_weight=None, verify=False):
        self._check_deterministic()
        self._fitted = False
        feature_blocks, n_samples, self._feature_schema, self.n_features_in_ = self._prepare_features(
            X_dense=X_dense, X_binary=X_binary, X_sparse=X_sparse, training=True
        )
        y, n_outputs = self._encode_target(y, n_samples)
        total_weight = float(n_samples)
        if sample_weight is not None:
            sample_weight = self._vector(sample_weight, n_samples)
            if sample_weight.min() < 0:
                raise ValueError("Negative sample weights")
            sample_weight = self._to_tensor(sample_weight, self.dtype)
            total_weight = float(sample_weight.sum())
            if not math.isfinite(total_weight) or total_weight <= 0:
                raise ValueError("Sample weights must have a finite positive sum")
        self.W_ = torch.zeros((self.n_features_in_, n_outputs), dtype=self.dtype, device=self.device)
        self.b_ = self.W_.new_zeros(n_outputs)
        regularization = (1 / self.C) / total_weight
        optimizer = torch.optim.LBFGS(
            [self.W_, self.b_], max_iter=self.max_iter, max_eval=self.max_eval,
            history_size=self.history_size, tolerance_grad=self.tol,
            tolerance_change=self.tolerance_change, line_search_fn="strong_wolfe")

        @torch.no_grad()
        def closure():
            loss, residual = self._loss_and_residual(self._logits(feature_blocks, n_samples), y, sample_weight)
            residual.div_(total_weight)
            weight_grad = torch.empty_like(self.W_)
            for feature_type, features, transpose, coef_slice in feature_blocks:
                gradient = weight_grad[coef_slice]
                if feature_type == "dense":
                    torch.mm(features.T, residual, out=gradient)
                elif feature_type == "sparse":
                    gradient.copy_(torch.sparse.mm(transpose, residual))
                else:
                    for column, indicator in enumerate(features):
                        gradient[column].copy_(torch.where(indicator[:, None], residual, 0.0).sum(dim=0))
            weight_grad.add_(self.W_, alpha=regularization)
            self.W_.grad, self.b_.grad = weight_grad, residual.sum(dim=0)
            return loss.div_(total_weight).add_(torch.dot(self.W_.flatten(), self.W_.flatten()), alpha=regularization / 2)

        optimizer.step(closure)
        state = optimizer.state[self.W_]
        self.n_iter_, self.n_eval_ = state["n_iter"], state["func_evals"]
        self.loss_ = self.grad_inf_norm_ = self.gradient_converged_ = None
        if verify:
            self.loss_ = float(closure())
            self.grad_inf_norm_ = max(float(self.W_.grad.abs().max()), float(self.b_.grad.abs().max()))
            if not math.isfinite(self.loss_) or not math.isfinite(self.grad_inf_norm_):
                raise FloatingPointError("Nonfinite objective or gradient")
            self.gradient_converged_ = self.grad_inf_norm_ <= self.tol
        self.W_.grad = self.b_.grad = None
        if not (torch.isfinite(self.W_).all() and torch.isfinite(self.b_).all()):
            raise FloatingPointError("Nonfinite fitted parameters")
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model is not fitted")

    def _predict_logits(self, X_dense, X_binary, X_sparse):
        self._check_fitted()
        self._check_deterministic()
        feature_blocks, n_samples, _, _ = self._prepare_features(
            X_dense=X_dense, X_binary=X_binary, X_sparse=X_sparse
        )
        return self._logits(feature_blocks, n_samples)

    @torch.no_grad()
    def decision_function(self, *, X_dense=None, X_binary=None, X_sparse=None):
        logits = self._predict_logits(X_dense, X_binary, X_sparse)
        return (logits if self.target_type == "categorical" else logits[:, 0]).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, *, X_dense=None, X_binary=None, X_sparse=None):
        logits = self._predict_logits(X_dense, X_binary, X_sparse)
        if self.target_type == "categorical":
            return logits.softmax(dim=1).cpu().numpy()
        probabilities = logits[:, 0].sigmoid_()
        if self.target_type == "continuous":
            return probabilities.cpu().numpy()
        result = probabilities.new_empty((probabilities.numel(), 2))
        result[:, 1].copy_(probabilities)
        torch.sub(1, probabilities, out=result[:, 0])
        return result.cpu().numpy()

    @torch.no_grad()
    def predict(self, *, X_dense=None, X_binary=None, X_sparse=None):
        logits = self._predict_logits(X_dense, X_binary, X_sparse)
        if self.target_type == "continuous":
            return logits[:, 0].sigmoid_().cpu().numpy()
        cutoff = math.log(self.threshold) - math.log1p(-self.threshold)
        class_indices = logits.argmax(dim=1) if self.target_type == "categorical" else (logits[:, 0] > cutoff).long()
        return self.classes_[class_indices.cpu().numpy()]

    @property
    def coef_(self):
        self._check_fitted()
        return self.W_.T.cpu().numpy().copy()

    @property
    def intercept_(self):
        self._check_fitted()
        return self.b_.cpu().numpy().copy()
