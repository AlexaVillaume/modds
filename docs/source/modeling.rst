========
Modeling
========

This module facilitates high-level Bayesian modeling of halo profiles.  Markov chain Monte Carlo sampling is currently provided with the `emcee<https://emcee.readthedocs.io/en/stable/>`_ package.

The :class:`~modeling.parameter.Parameter` class keeps track of the prior distributions and parameter transformations for free parameters in the model.  Prior probability distributions can either be taken from those in `scipy.stats<https://docs.scipy.org/doc/scipy/reference/stats.html>`_ or constructed by hand.

The :class:`~modeling.measurement.MeasurementModel` class wraps up a parameterized halo mass profile (a subclass of :class:`~halo.profile_base.HaloDensityProfile`), a list of free :class:`~modeling.parameter.Parameter` instances, and a set of observables (e.g., density, mass or surface density as a function of radius).  Once created, a `MeasurementModel` instance is callable, and maps a list of parameter values to a log posterior probability.

The :mod:`~modeling.h5io` module contains utilities for managing `hdf5` files.  In particular, it provides the :class:`~modeling.h5io.HDF5Backend` class, which manages a saved model and the results of sampling from the model.

The installed `pfit.py` script wraps all this together in a nice command line interface.

----------------
Module reference
----------------

.. toctree::
    :maxdepth: 3

    modeling_example
    modeling_measurement
    modeling_parameter
    modeling_h5io
