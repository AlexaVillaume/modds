"""
joint_priors.py

Joint priors from halo scaling relations.
"""

import numpy as np

from colossus.lss.peaks import peakHeight
from colossus.halo.concentration import concentration

from modds.measurement import lnlike_gauss

def alpha_nu_gao08(profile, **kwargs):
    """log normal distribution of alpha about the
    alpha--peak height relation from Gao+2008"""
    
    # scatter in dex
    if "sigma_alpha" in kwargs:
        sigma_alpha = kwargs["sigma_alpha"]
    else:
        sigma_alpha = 0.1
    z = kwargs["z"]

    M = profile.MDelta(z, "vir")
    nu = peakHeight(M, z)
    alpha_model = 0.155 + 0.0095 * nu**2

    return lnlike_gauss(np.log10(alpha_model), np.log10(alpha), sigma_alpha)


def hmcr_dk15(profile, **kwargs):
    """log normal distribution for concentration
    about the halo mass--concentration relation of Diemer & Kravtsov 2015
    """
    
    # scatter in dex
    if "sigma_c" in kwargs:
        sigma_c = kwargs["sigma_c"]
    else:
        sigma_c= 0.16
    z = kwargs["z"]

    try:
        rvir, Mvir = profile.RMDelta(z, "vir")
        log_cvir_model = np.log10(concentration(M=Mvir, z=z, mdef="vir"))
    except:
        # can't find concentration, reject model
        return -np.inf
    
    rs = kwargs["rs"]
    # should match...
    assert rs == profile.par["rs"]
    log_cvir = np.log10(rvir / rs)

    return lnlike_gauss(log_cvir_model, log_cvir, sigma_c)
    
