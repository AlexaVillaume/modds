"""
measurement.py

MeasurementModel wraps up a halo.HaloDensityProfile instance with both a set of 
observables (e.g., {r, DeltaSigma, DeltaSigma_err}) and a prior on the model 
parameters of interest.
"""

from collections import OrderedDict

import numpy as np
from scipy import optimize
from colossus.halo.profile_base import HaloDensityProfile

from modds.parameter import Parameter

__all__ = ['MeasurementModel']


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
        Map from constant name (str) to fixed physical value (float).
        Should contain redshift (named "z") if not defined as a parameter
    lnlike : callable, optional
        Map from (q_model, q, q_err) to log likelihood at each point.  Defaults
        to a Gaussian likelihood, i.e., q ~ N(q_model, q_err**2)
    priors : iterable, optional
        List of functions, f(profile, **kwargs) -> log prior.
        Additional priors to consider (e.g., halo mass concentration relation,
        See examples in joint_priors.py.
        Keywords should be either constants or parameters of the model.
    """

    # mapping of inputtable quantities to colossus halo profile function name
    _quantities = dict(rho='density', density='density', m='enclosedMass',
                       mass='enclosedMass', enclosedmass='enclosedMass',
                       sigma='surfaceDensity', surfacedensity='surfaceDensity',
                       deltasigma='deltaSigma')
    _obskeys = ['r', 'q', 'q_err']

    def __init__(self, profile, observables, quantity, parameters,
                 constants=None, lnlike=lnlike_gauss, priors=None):
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
        self.priors = priors
        assert ("z" in self.constants) or ("z" in self.parameters)

        
    def _get_required_parameters(self, sample):
        """The parameters the sampler sees are different than what colossus 
        needs.  This glues the two together.  `sample` is what the sampler 
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
                    new_pars[i] = sample[idx]
                else:
                    # need to do the inverse transform to physical values
                    new_pars[i] = p.inverse_transform(sample[idx])
        return new_pars

    
    def _get_kwargs(self, sample):
        """Construct a dictionary of keyword arguments from a point in sample
        space.  Includes constant values.
        """
        kwargs = {}
        for i, (name, p) in enumerate(self.parameters.items()):
            if p.transform is not None:
                kwargs[name] = p.inverse_transform(sample[i])
            else:
                kwargs[name] = sample[i]
        kwargs = {**kwargs, **self.constants}
        return kwargs

    
    def update(self, sample):
        """Update the profile with the passed values.

        Parameters
        ----------
        sample : array_like
            Size of ndim, values as the sampler would see them (i.e., 
            transformed)        

        Returns
        -------
        bool
            True if successful
        """
        # set new profile parameters and update
        
        new_pars = self._get_required_parameters(sample)
        if 'z' in self.parameters:
            # update redshift with new value
            p = self.parameters['z']
            idx = list(self.parameters.keys()).index('z')
            if p.transform is not None:
                z = p.inverse_transform(sample[idx])
            else:
                z = sample[idx]
            self.profile.opt['z'] = z
        else:
            assert self.profile.opt['z'] == self.constants['z']
        
        self.profile.setParameterArray(new_pars)
        try:
            self.profile.update()
            return True
        except:
            # TODO: catch the specific exception
            # handle case where the halo density is too small
            return False

    def __call__(self, sample,
                 return_lnlike=False,
                 return_profile=False,
                 return_vir=False,
                 return_sp=False,
                 r_grid=None,
                 mdef="vir",
                 log_rsp_search_min=2.5,
                 log_rsp_search_max=3.5):
        """
        Calculate the log posterior probability for the model.

        Parameters
        ----------
        sample : array_like
            length ndim array of transformed parameters
        return_lnlike : bool, optional
            if True, also return the log likelihood
        return_profile : bool, optional
            if True, also return the model interpolated on the specified grid
        return_vir : bool, optional
            if True, also return the halo virial mass and concentration
        return_sp : bool, optional
            if True, also return the halo splashback radius and steepest slope
        r_grid : array_like, optional
            Radial interpolation grid for when caching the posterior prediction
        mdef : str, optional
            halo virial mass definition
        log_rsp_search_min : float, optional
            log of kpc, minimum radius to search for rsp
        log_rsp_search_max : float, optional
            log of kpc, maximum radius to search for rsp

        Returns
        -------
        lnpost : float
            log of posterior probability
        blobs : tuple
            tuple of optional returns, ordered as (lnlike, profile, Mvir, cvir, 
            rsp, gamma_min)
        """
        return_blobs = np.any([return_lnlike, return_profile, return_vir,
                               return_sp])
        
        # update profile with new values
        successful_update = self.update(sample)
        
        # calculate log prior, returning early on bad values
        lnp = 0
        for i, p in enumerate(self.parameters.values()):
            lnp += p(sample[i])
        if self.priors is not None:
            for prior in self.priors:
                lnp += prior(self.profile, **self._get_kwargs(sample))
        if not np.isfinite(lnp) or not successful_update:
            if return_blobs:
                # construct rejected blobs
                blobs = []
                if return_lnlike:
                    blobs.append(np.nan)
                if return_profile:
                    blobs.append(np.nan * np.ones(r_grid.shape))
                if return_vir:
                    blobs.append(np.nan)
                    blobs.append(np.nan)
                if return_sp:
                    blobs.append(np.nan)
                    blobs.append(np.nan)
                return -np.inf, tuple(blobs)

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
                q_model = np.nan * r
        lnl = np.sum(self.lnlike(q_model, q, q_err))

        if return_blobs:
            kwargs = self._get_kwargs(sample)
            blobs = []
            if return_lnlike:
                blobs.append(lnl)
            if return_profile:
                q_grid = np.interp(r_grid, r, q_model)                
                blobs.append(q_grid)
            if return_vir:
                z = kwargs['z']
                rvir, Mvir = self.profile.RMDelta(z=z, mdef=mdef)
                rs = kwargs['rs']
                cvir = rvir / rs
                blobs.append(Mvir)
                blobs.append(cvir)
            if return_sp:
                rsp = optimize.fminbound(self.profile.densityDerivativeLog,
                                         10**log_rsp_search_min,
                                         10**log_rsp_search_max)
                gamma_min = self.profile.densityDerivativeLog(rsp)
                blobs.append(rsp)
                blobs.append(gamma_min)
            return lnp + lnl, tuple(blobs)
        else:
            return lnp + lnl
