from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.stats as stat

# This part of code is adapted from the following git:
# https://gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d


class LogisticReg:

    def __init__(self, *args, **kwargs):
        self.model = LogisticRegression(*args, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

        denom = (2.0*(1.0+np.cosh(self.model.decision_function(X))))
        F_ij = np.dot((X / denom[:, None]).T, X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.array([np.sqrt(Cramer_Rao[i, i]) for i in
                                   range(Cramer_Rao.shape[0])])
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]

        self.z_scores = z_scores
        self.p_values = np.array(p_values)
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij
