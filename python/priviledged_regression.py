from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from pymks import PrimitiveBasis


class PriviledgedLinearRegression(LinearRegression):

    def __init__(self, domain=[-100, 1000], dx=0.1,
                 eps=1e-10, temp=3., *args, **kwargs):
        self._teacher = LinearRegression()
        self._student = LinearRegression()
        self._temp = temp
        self._lambda = 0.25
        self.domain = domain
        self.eps = eps
        self.dx = dx
        super(PriviledgedLinearRegression, self).__init__(*args, **kwargs)

    def fit(self, X, X_priv, y):
        self._teacher.fit(X_priv, y)
        self._student.fit(X, y)
        n_coeffs = self._student.coef_.shape[-1]
        x0 = [self._lambda] + self._student.coef_.tolist()[0] +\
            self._student.intercept_.tolist()
        print x0
        bounds = [(None, None) for i in self._student.coef_.tolist()[0]]
        bounds = [(0.0, 0.65), (None, None)] + bounds
        res = minimize(self._minization_function, x0, args=(X, y, n_coeffs),
                       method='L-BFGS-B', bounds=bounds)
        if res.success:
            print(res.x)
            print(res.message)
            self._student.intercept_ = res.x[n_coeffs + 1]
            self._student.coef_ = np.array([i for i in
                                            res.x[1:n_coeffs + 1]])[None]

    def _minization_function(self, vars, X, y, n_coeffs):
        self._student.coef_ = np.array([i for i in vars[1:n_coeffs + 1]])[None]
        self._student.intercept_ = vars[n_coeffs + 1]
        y_soft = self._continuous_softmax(y, self._temp)
        y_pred = self._student.predict(X)
        y_pred_soft = self._continuous_softmax(y_pred, 1)
        tmp = ((1 - vars[0]) * self._kl_divergence(y, y_pred_soft) +
               vars[0] * self._kl_divergence(y_soft, y_pred_soft))
        return tmp

    def _kl_divergence(self, y, y_pred):
        return np.sum(y * (np.log(y + self.eps) - np.log(y_pred + self.eps)))

    def _continuous_softmax(self, y, T):
        y_dist = self._property2distribution(y)
        return gaussian_filter1d(y_dist, T)

    def predict(self, X):
        return self._student.predict(X)

    def _property2distribution(self, y):
        n_bins = int((self.domain[1] - self.domain[0] + self.dx) / self.dx)
        p_basis = PrimitiveBasis(n_states=n_bins, domain=self.domain)
        return p_basis.discretize(y)[:, 0, :]
