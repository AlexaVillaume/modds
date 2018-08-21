.. modds documentation master file, created by
   sphinx-quickstart on Sat Aug 18 10:37:44 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================
ModDS Documentation
=================================

ModDS is a code for specifying and sampling from Bayesian models of dark matter halos.  It uses `colossus <https://bdiemer.bitbucket.io/colossus/>`_ for physical halo models and `emcee <https://emcee.readthedocs.io/en/stable/>`_ for Markov chain Monte Carlo sampling.

The :class:`~modds.parameter.Parameter` class keeps track of the prior distributions and parameter transformations for free parameters in the model.  Prior probability distributions can either be taken from those in `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ or constructed by hand.

The :class:`~modds.measurement.MeasurementModel` class wraps up a parameterized halo mass profile (a subclass of :class:`colossus.halo.profile_base.HaloDensityProfile`), a list of free :class:`~modds.parameter.Parameter` instances, and a set of observables (e.g., density, mass or surface density as a function of radius).  Once created, a `MeasurementModel` instance is callable, and maps a list of parameter values to a log posterior probability.

The :mod:`~modds.h5io` module contains utilities for managing `hdf5` files.  In particular, it provides the :class:`~modds.h5io.HDF5Backend` class, which manages a saved model and the results of sampling from the model.

The installed `modds` script wraps all this together in a nice command line interface.

.. toctree::
   :maxdepth: 2
	      
   installation
   api
   example

	     



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
