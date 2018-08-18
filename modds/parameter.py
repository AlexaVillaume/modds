"""
parameter.py

The Parameter class wraps up a prior probability distribution from scipy.stats
with an optional parameter transformation.
"""

import numpy as np
from scipy import stats

__all__ = ['Parameter']


class Parameter():
    """
    A model parameter with a prior.

    Often we want to sample over some transformation of the parameter, e.g., 
    sampling with a uniform distribution over the log of some scale parameter.
    Thus the user can specify a transformation to map between the value as seen
    by the physical functions and the value as seen by the sampling algorithm.

    Priors can be specified from those built into scipy.stats.
    See the list here:
    https://docs.scipy.org/doc/scipy/reference/stats.html

    You can either directly pass a frozen scipy.stats.rv_continous instance, or 
    a string of the form "prior(loc=0,scale=1)" where "prior" is the name of the
    distribution, and "loc"/"scale" represent location and scale parameters of 
    the distribution.  For example, to specify a Gaussian prior with a mean of
    0 and a standard deviation of 1, you would pass "norm(loc=0, scale=1)", 
    while for a uniform distribution between 3 and 10, you would pass
    "uniform(loc=3, scale=7)".  Other distributions may have more complicated 
    call signatures.  See the scipy documentation for more details.

    Parameters
    ----------
    name : str
    prior : str or stats._distn_infrastructure.rv_frozen instance
        Prior probability distribution.  If a string is given, it
        should be of the form "prior(a=1.2, b=3.4) where "prior" is some 
        built-in prior model and "a"/"b" are the prior parameters (e.g., loc and
        scale). If an instance of scipy.stats.rv_continous is passed, the 
        parameters should be frozen-in (i.e., you can call it without specifying
        hyper-parameters).
    transform : callable or str, optional
        A map from the parameter as seen by the sampler to the parameter as seen
        by colossus (i.e., the physical value).  If a string is given, it should
        be one of the built-in transforms.  Currently these are
        "from_log" : transform from the base-10 logarithm
    inverse_transform : callable, optional
        If a user-specified transformation is given instead of a built-in one, 
        then you must also pass in the inverse transformation from the physical
        parameter to the parameter as seen by the sampler.  E.g., for the log
        transform, the inverse_transform could be given as "lambda x: 10**x".
    """

    # dictionary of recognized transformations and inverses
    _transforms = {'from_log': (lambda x: np.log10(x),
                                lambda x: 10**x)}

    def __init__(self, name, prior, transform=None, inverse_transform=None):

        self.name = name

        # parse prior
        if isinstance(prior, stats._distn_infrastructure.rv_frozen):
            self.prior = prior
            self.prior_str = None
        else:
            # string wrangling to create a frozen stats.rv_continuous instance
            prior = prior.replace(' ', '')
            self.prior_str = prior
            # split into names and hyper parameters
            prior_name, hp = prior.split('(')
            # fudge hp string into dict of keywords
            hp = hp.strip(')').split(',')
            hp = [(s.split('=')[0], float(s.split('=')[1])) for s in hp]
            hp = dict(hp)
            # attempt to construct a distribution from scipy.stats
            try:
                dist = getattr(stats, prior_name)
            except AttributeError:
                raise ValueError("{} was not found in "
                                 "scipy.stats!".format(prior_name))
            # freeze in hyper-parameters
            self.prior = dist(**hp)

        # parse transform
        if transform is None:
            self.transform = None
            self.inverse_transform = None
        elif transform in self._transforms:
            self.transform, self.inverse_transform = self._transforms[transform]
        else:
            # should be callable and have a defined inverse
            self.transform = transform
            self.inverse_transform = inverse_transform
            # spot check that...
            x = self.prior.rvs(size=50)
            if not np.all(np.isclose(x, inverse_transform(transform(x)))):
                raise ValueError(
                    "transform and inverse_transform don't match!")

    def __call__(self, x):
        """
        Calculate the log prior at the given value.

        Parameters
        ----------
        x : float or array_like
             value of the parameter as seen by the sampler (i.e., the 
             transformed value)

        Returns
        -------
        float or array_like
            The log of the prior probability distribution at the given value(s)
        """
        return self.prior.logpdf(x)

    def __repr__(self):
        if self.prior_str is not None:
            return "<Parameter: {} ~ {}>".format(self.name, self.prior_str)
        else:
            return "<Parameter: {}>".format(self.name)

    def sample(self, n):
        """
        Draws n samples from the prior distribution.

        Parameters
        ----------
        n : int

        Returns
        -------
        array_like
        """
        return self.prior.rvs(size=n)
