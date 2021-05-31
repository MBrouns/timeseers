from abc import ABC, abstractmethod
import pymc3 as pm


class Likelihood(ABC):
    """Subclasses should implement the observed method which defines an observed random variable"""
    @property
    def distribution(self):
        raise NotImplementedError

    def observed(self, mu, y_scaled, sample_weight):
        model_parameters = self._likelihood_priors()

        model_parameters["mu"] = mu

        if sample_weight is None:
            self.distribution("obs", **model_parameters, observed=y_scaled)
        else:
            pm.Potential("obs", sample_weight * self.distribution.dist(**model_parameters).logp(y_scaled))

    @abstractmethod
    def _likelihood_priors(self):
        """
        Dictionary of priors for the observation likelihood. Excludes `mu`
        """
        pass


class Gaussian(Likelihood):
    """Gaussian likelihood with constant variance"""
    distribution = pm.Normal

    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def _likelihood_priors(self):
        return {
            "sigma": pm.HalfCauchy("sigma", self.sigma)
        }


class StudentT(Likelihood):
    """StudentT likelihood with constant variance, robust to outliers"""
    distribution = pm.StudentT

    def __init__(self, alpha=1., beta=1., sigma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    def _likelihood_priors(self):
        return {
            "nu": pm.InverseGamma("nu", alpha=self.alpha, beta=self.beta),
            "sigma": pm.HalfCauchy("sigma", self.sigma)
        }
