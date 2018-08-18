"""
measurement.py

MeasurementModel wraps up a halo.HaloDensityProfile instance with both a set of 
observables (e.g., {r, DeltaSigma, DeltaSigma_err}) and a prior on the model 
parameters of interest.
"""

from collections import OrderedDict
import numpy as np

from colossus.halo.profile_base import HaloDensityProfile
from modds.parameter import Parameter

__all__ = ['MeasurmentModel']

def lnlike_gauss(q_model, q, q_err):
    """
    Log of Gaussian likelihood.

    Parameters
    ----------
    q_model : array_like
        predicted value
    q : array_like
        observed value
    q_err : array_like
        observational uncertainty

    Returns
    -------
    lnl : array_like
        log likehood array, same size as input observations
    """
    var = q_err**2
    return -0.5 * (np.log(2 * np.pi * var) + (q - q_model)**2 / var)


class MeasurementModel():
    """
    A wrapped up halo density profile with data constraints and priors.

    Parameters
    ----------
    profile : halo.profile_bass.HaloDensityProfile
    observables : dict or iterable
        Should have keys of 'r', 'q', 'q_err', else it is assumed to be an
        iterable containing three arrays in the order r, q, q_err.
    quantity : str
        One of {'rho', 'M', 'Sigma', 'DeltaSigma'}, with standard colossus 
        units.  Denotes volume mass density, enclosed mass, surface density,
        and surface density deviation (i.e., the weak lensing observable)
        respectively.
    parameters : list of colossus.modeling.Parameter instances
    constants : dict
        Map from constant name (str) to fixed physical value (float)
    lnlike : callable, optional
        Map from (q_model, q, q_err) to log likelihood at each point.  Defaults
        to a Gaussian likelihood, i.e., q ~ N(q_model, q_err**2)
    """

    # mapping of inputtable quantities to colossus halo profile function name
    _quantities = dict(rho='density', density='density', m='enclosedMass',
                       mass='enclosedMass', enclosedmass='enclosedMass',
                       sigma='surfaceDensity', surfacedensity='surfaceDensity',
                       deltasigma='deltaSigma')
    _obskeys = ['r', 'q', 'q_err']
    
    def __init__(self, profile, observables, quantity, parameters,
                 constants=None, lnlike=lnlike_gauss):
        # check this is an actual halo density profile
        assert isinstance(profile, HaloDensityProfile)
        self.profile = profile
        # check this is an allowed quantity
        quantity = quantity.replace(' ', '').lower()
        assert quantity in self._quantities
        self.quantity = self._quantities[quantity]
        # construct ordered dict of observables
        if isinstance(observables, OrderedDict):
            assert all([key in observables for key in self._obskeys])
            self.observables = observables
        elif isinstance(observables, dict):
            assert all([key in observables for key in self._obskeys])
            self.observables = OrderedDict([(key, observables[key])
                                            for key in self._obskeys])
        else:
            self.observables = OrderedDict(zip(self._obskeys, observables))
        # check that everything is a proper parameter
        assert all([isinstance(p, Parameter) for p in parameters])
        self.parameters = OrderedDict([(p.name, p) for p in parameters])
        self.ndim = len(parameters)
        # set default constants to empty dictionary
        if constants is None:
            constants = {}
        self.constants = constants
        self.lnlike = lnlike

    def _get_required_parameters(self, values):
        """The parameters the sampler sees are different than what colossus 
        needs.  This glues the two together.  `values` are what the sampler 
        sees.
        """
        # construct array for profile prediction
        new_pars = np.zeros(len(self.profile.par))
        for i, required_param in enumerate(self.profile.par_names):
            try:
                new_pars[i] = self.constants[required_param]
            except KeyError:
                # must be a free parameter, transform if need be
                p = self.parameters[required_param]
                # index into values for free parameters
                idx = list(self.parameters.keys()).index(required_param)
                if p.transform is None:
                    new_pars[i] = values[idx]
                else:
                    # need to do the inverse transform to physical values
                    new_pars[i] = p.inverse_transform(values[idx])
        return new_pars
    
    def update(self, values):
        """Update the profile with the passed values.

        Parameters
        ----------
        values : array_like
            Size of ndim, values as the sampler would see them (i.e., 
            transformed)        

        Returns
        -------
        bool
            True if successful
        """
        # set new profile parameters and update
        new_pars = self._get_required_parameters(values)
        self.profile.setParameterArray(new_pars)
        try:
            self.profile.update()
            return True
        except:
            #TODO: catch the specific exception
            # handle case where the halo density is too small
            return False
        
        
    def __call__(self, values, return_lnlike=False, return_model=False,
                 r_grid=None):
        """
        Calculate the log posterior probability for the model.

        Parameters
        ----------
        values : array_like
            length ndim array of transformed parameters
        return_lnlike : bool, optional
            if True, also return the log likelihood
        return_model : bool, optional
            if True, also return the model interpolated on the specified grid
        r_grid : array_like, optional
            Radial interpolation grid for when caching the posterior prediction

        Returns
        -------
        lnpost : float
            log of posterior probability
        lnl : float, optional
            log likelihood, only returned if return_lnlike is True
        model_grid : array_like, optional
            posterior predicted observable at the r_grid interpolation points
        """
        # update profile with new values
        successful_update = self.update(values)
        
        # calculate log prior, returning early on bad values
        lnp = 0
        for i, p in enumerate(self.parameters.values()):
            lnp += p(values[i])
        if not np.isfinite(lnp) or not successful_update:
            # handle different blob size returns
            if return_lnlike and return_model:
                return -np.inf, (None, None)
            elif return_lnlike or return_model:
                return -np.inf, (None,)
            else:
                return -np.inf
            
        # calculate log likelihood
        r = self.observables['r']
        q = self.observables['q']
        q_err = self.observables['q_err']
        try:
            q_model = getattr(self.profile, self.quantity)(r)
        except:
            # TODO: selectively catch interpolation failure
            # try with interpolation off
            try:
                kwargs = dict(interpolate=False,
                              interpolate_surface_density=False)
                q_model = getattr(self.profile, self.quantity)(r, **kwargs)
            except:
                # and fail somewhat gracefully if that doesn't work
                q_model = -np.inf * r
        lnl = np.sum(self.lnlike(q_model, q, q_err))

        if return_model:
            q_grid = np.interp(r_grid, r, q_model)
            if return_lnlike:
                return lnp + lnl, (lnl, q_grid)
            else:
                return lnp + lnl, (q_grid,)
        else:
            if return_lnlike:
                return lnp + lnl, (lnl,)
            else:
                return lnp + lnl


