import numpy as np
import torch
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from logistic_regression_onehot_v2 import LogisticRegression


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=False)
    rng = np.random.default_rng(9)
    n = 1800
    d = rng.normal(size=(n, 3))
    b = rng.random((n, 2)) < .4
    s = sparse.csr_matrix(rng.random((n, 4)) < .13)
    one_hot = np.column_stack((np.eye(4, dtype=bool)[rng.integers(0, 4, n)],
                               np.eye(3, dtype=bool)[rng.integers(0, 3, n), :2]))
    b = np.column_stack((b, one_hot))
    a = np.column_stack((d, b, s.toarray()))
    blocks = dict(X_dense=d, X_binary=b, X_sparse=s)
    fit_blocks = blocks
    sw = rng.uniform(.2, 2, n)
    for kind, y in [('binary', rng.integers(0, 2, n)), ('categorical', rng.integers(0, 3, n))]:
        for weight in [None, sw]:
            kwargs = dict(target_type=kind, C=.3, tol=1e-10, max_iter=300)
            m = LogisticRegression(**kwargs).fit(y=y, sample_weight=weight, verify=True, **fit_blocks)
            m2 = LogisticRegression(**kwargs).fit(y=y, sample_weight=weight, **fit_blocks)
            assert np.array_equal(m.coef_, m2.coef_)
            assert np.array_equal(m.intercept_, m2.intercept_)
            assert np.array_equal(m.predict_proba(**blocks), m2.predict_proba(**blocks))
            ref = SkLogisticRegression(C=.3, tol=1e-12, max_iter=1000).fit(a, y, sample_weight=weight)
            err = np.max(np.abs(m.predict_proba(**blocks) - ref.predict_proba(a)))
            assert err < 3e-6, err
            assert np.array_equal(m.predict(**blocks), ref.predict(a))
            print(kind, 'weighted' if weight is not None else 'unweighted', 'max probability error', err, 'repeat exact', 'gradient', m.grad_inf_norm_)
    y = rng.uniform(0, 1, n)
    m = LogisticRegression('continuous', C=.3, tol=1e-10, max_iter=300).fit(y=y, sample_weight=sw, **fit_blocks)
    p = a.shape[1]
    def obj(v):
        z = a @ v[:p] + v[-1]
        r = sw * (expit(z) - y) / sw.sum()
        loss = np.dot(sw, np.logaddexp(0, z) - y*z)/sw.sum() + np.dot(v[:p], v[:p])/(2*.3*sw.sum())
        return loss, np.r_[a.T @ r + v[:p]/(.3*sw.sum()), r.sum()]
    ref = minimize(obj, np.zeros(p+1), jac=True, method='L-BFGS-B', options={'gtol':1e-12,'ftol':1e-15})
    err = np.max(np.abs(m.predict(**blocks)-expit(a @ ref.x[:p]+ref.x[-1])))
    assert err < 3e-6
    print('fractional weighted max probability error', err)
    for features in (dict(X_dense=d), dict(X_binary=b), dict(X_sparse=s)):
        m = LogisticRegression().fit(np.arange(n)%2, **features)
        assert m.predict_proba(**features).shape == (n,2)
    y_binary = np.arange(n) % 2
    dense_onehot = LogisticRegression(tol=1e-10).fit(y_binary, X_binary=b)
    sparse_onehot = LogisticRegression(tol=1e-10).fit(y_binary, X_sparse=sparse.csr_matrix(b))
    np.testing.assert_allclose(dense_onehot.predict_proba(X_binary=b),
                               sparse_onehot.predict_proba(X_sparse=sparse.csr_matrix(b)), atol=1e-7, rtol=0)
    try:
        LogisticRegression().fit(np.arange(n)%2, X_dense=d, sample_weight=np.zeros(n))
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted all-zero weights')
    print('single feature blocks and invalid weights passed')

    # Explicit precision regression tests with identical mixed feature blocks.
    for dtype, np_dtype, tol, limit in [(torch.float64, np.float64, 1e-10, 3e-6),
                                        (torch.float32, np.float32, 1e-6, 2e-3)]:
        for kind, y in [('binary', rng.integers(0, 2, n)),
                        ('categorical', rng.integers(0, 3, n)),
                        ('continuous', rng.uniform(0, 1, n))]:
            for weight in (None, sw):
                kwargs = dict(target_type=kind, C=.3, tol=tol, max_iter=300, dtype=dtype)
                m = LogisticRegression(**kwargs).fit(y=y, sample_weight=weight, verify=True, **fit_blocks)
                m2 = LogisticRegression(**kwargs).fit(y=y, sample_weight=weight, **fit_blocks)
                pred = m.predict_proba(**blocks)
                assert m.coef_.dtype == m.intercept_.dtype == pred.dtype == np_dtype
                assert m.decision_function(**blocks).dtype == np_dtype
                assert np.array_equal(m.coef_, m2.coef_)
                assert np.array_equal(m.intercept_, m2.intercept_)
                assert np.array_equal(pred, m2.predict_proba(**blocks))
                if kind == 'continuous':
                    def objective(v):
                        z = a @ v[:-1] + v[-1]
                        w = np.ones(n) if weight is None else weight
                        r = w * (expit(z) - y) / w.sum()
                        loss = np.dot(w, np.logaddexp(0, z) - y*z)/w.sum() + np.dot(v[:-1], v[:-1])/(2*.3*w.sum())
                        return loss, np.r_[a.T @ r + v[:-1]/(.3*w.sum()), r.sum()]
                    ref = minimize(objective, np.zeros(a.shape[1]+1), jac=True, method='L-BFGS-B', options={'gtol':1e-12,'ftol':1e-15})
                    expected = expit(a @ ref.x[:-1]+ref.x[-1])
                else:
                    ref = SkLogisticRegression(C=.3, tol=1e-12, max_iter=1000).fit(a, y, sample_weight=weight)
                    expected = ref.predict_proba(a)
                err = np.max(np.abs(pred-expected))
                assert err < limit, err
                print(dtype, kind, 'weighted' if weight is not None else 'unweighted', 'max probability error', err, 'repeat exact')
    try:
        LogisticRegression(dtype=torch.float16)
    except ValueError:
        pass
    else:
        raise AssertionError('Unsupported dtype accepted')

    def expect_error(error_type, operation):
        try:
            operation()
        except error_type:
            return
        raise AssertionError(f"Expected {error_type.__name__}")

    for target_type, target in (("binary", y_binary),
                                ("continuous", rng.uniform(0, 1, n)),
                                ("categorical", np.arange(n) % 3)):
        expect_error(ValueError, lambda: LogisticRegression(target_type).fit(target[:, None], X_dense=d))
    expect_error(TypeError, lambda: LogisticRegression().fit(y_binary, X=d))
    expect_error(ValueError, lambda: LogisticRegression().fit(y_binary, X_binary=b.astype(np.uint8)))
    expect_error(ValueError, lambda: LogisticRegression().fit(y_binary, X_dense=d, X_binary=b[:-1]))
    expect_error(ValueError, lambda: LogisticRegression().fit(y_binary, X_sparse=s.tocoo()))
    noncanonical = sparse.csr_matrix((np.ones(2), np.array([0, 0]), np.array([0, 2])), shape=(1, 1))
    expect_error(ValueError, lambda: LogisticRegression().fit(np.array([0]), X_sparse=noncanonical))
    m = LogisticRegression().fit(y_binary, **blocks)
    expect_error(ValueError, lambda: m.predict_proba(X_dense=d[:, :-1], X_binary=b, X_sparse=s))
    expect_error(ValueError, lambda: m.predict_proba(X_dense=d, X_binary=b[:, :-1], X_sparse=s))
    expect_error(ValueError, lambda: m.predict_proba(X_dense=d, X_binary=b, X_sparse=s[:, :-1]))
    expect_error(ValueError, lambda: m.predict_proba(X_dense=d, X_sparse=s))
    expect_error(TypeError, lambda: m.predict_proba(X=d))
    class TargetValidationOnly(LogisticRegression):
        @staticmethod
        def _check_numeric(values):
            assert values.ndim == 1
            assert values.shape == (n,)
            LogisticRegression._check_numeric(values)
    TargetValidationOnly().fit(y_binary, **blocks)
    print('Explicit inputs, one-dimensional targets, structural checks and removed feature scans passed')

if __name__ == '__main__':
    main()
