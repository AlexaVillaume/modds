======================
Example model sampling
======================

Suppose we have weak lensing data, (radial samples, :math:`R`, relative surface density, :math:`\\Delta\\Sigma`, and uncertainty, :math:`\\delta\\Delta\\Sigma`) that we wish to model with an `Einasto profile<https://en.wikipedia.org/wiki/Einasto_profile>`_.  This model has three parameters (:math:`\\rho_s`, :math:`\\r_s`, :math:`\\alpha`).  To ensure that this parameters remain positive, we will model them with a wide uniform prior over the logarithm of these parameters.

.. math::

   \\log_{10} \\rho_s \sim \mathcal{U}(4, 9)
   \\log_{10} \\r_s \sim \mathcal{U}(0, 3)
   \\log_{10} \\alpha \sim \mathcal{U}(-0.7, 0.1)

   
where the units are :math:`M_\odot h^2 \ \mathrm{kpc}^{-3}` and :math:`\mathrm{kpc} h^{-1}` for :math:`\\rho_s` and :math:`\\r_s` respectively.

To setup this model, we can write out our configuration in a `yaml<http://yaml.org/start.html>`_ file.  For the purposes of using `pfit.py`, you can think of this as a python dictionary in file form.

.. yaml::

   output: mymodel.hdf5
   
   model:
     profile: Einasto
     parameters:
     - rhos:
         prior: uniform(loc=4, scale=5)
         transform: from_log
     - rs:
         prior: uniform(loc=0, scale=3)
         transform: from_log
     - alpha:
         prior: norm(loc=-0.7, scale=0.1)
         transform: from_log
     quantity: deltaSigma
     observables: mydata.csv
	       
   settings:
     z: 0.5
     cosmo: planck15
     nwalkers: 64
     r_grid: 50
   

The first key specifies the name of the output file.  The second key, ``model``, specifies the configuration of the model.  In particular, we can choose our profile from any of those recognized by `colossus`.

The parameters of the model are specified as a list (hence the hyphens), as the order needs to be fixed.  An individual free parameter needs to have its prior probability distribution specified with a string that can be interpreted as a `scipy.stats continuous distribution<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_.  This format is simply the name of the distribution with parenthesis around the fixed hyper-parameters of the distribution.  For most common distributions, these are a location (``loc``) and scale (``scale``) parameter.  For a Gaussian distribution, this corresponds to the mean and standard deviation.  For a uniform distribution, this corresponds to the lower bound and the difference between the lower and upper bounds.  More details can be found in the `scipy.stats documentation<https://docs.scipy.org/doc/scipy/reference/stats.html>`_.

Parameters can also have an optional transform.  The idea here is that the space on which we want to sample the model (e.g., the logarithm of the parameters in this example) is not necessarily the space in which we want to evaluate our physical models (i.e., `colossus` expects physical densities, not the logarithm of physical densities).  Currently only a logarithmic transform is implemented, but users can construct parameters with transforms and inverse transforms by directly instantiating the :class:`~modeling.parameter.Parameter` class if need be.

The ``quantity`` key should specify what physical quantity is being modeled, and can be any quantity that a :class:`~halo.profile_base.HaloDensityProfile` instance can model, e.g., ``rho``, ``sigma``, ``M``, ``DeltaSigma``.

``observables`` should either be a ``*.csv`` filename or a dictionary of arrays with keys of ``r``, ``q``, and ``q_err`` (radii, quantities, and observational uncertainty respectively).  If a filename is passed, the file should be comma-delimited with rows of measurements and columns of (``r``, ``q``, ``q_err``).

The ``settings`` dictionary carries a few necessary settings.  ``z`` is the redshift corresponding to the dataset, and it currently must be fixed.  ``cosmo`` represent the adopted cosmology and should be a string recognized by :function:`~cosmology.cosmology.setCosmology`.  ``nwalkers`` is the number of parameter walkers used in the Goodman & Weare ensemble sampler.  It should be an even integer, and at very least twice the number of free parameters.  ``r_grid`` is the number of grid points for the stored posterior predicitive checks.

If we wanted to keep one of the parameters in the model fixed, we would add that fixed value to a dictionary of constants, e.g.,

.. yaml::

   output: mymodel.hdf5
   
   model:
     profile: Einasto
     parameters:
     - rhos:
         prior: uniform(loc=4, scale=5)
         transform: from_log
     - rs:
         prior: uniform(loc=0, scale=3)
         transform: from_log
     constants:
       alpha: 0.2
     quantity: deltaSigma
     observables: test_data.csv
	       
   settings:
     z: 0.5
     cosmo: planck15
     nwalkers: 64
     r_grid: 50


We can initialize an output file from this configuration with the ``init`` command, e.g. `pfit.py init mymodel.yaml`.  This will create the ``mymodel.hdf5`` file.  We can then sample from this model with `pfit.py sample mymodel.hdf5 1000` where the second argument to sample indicates that we will sample for 1000 iterations.  We can optionally make use of multiple threads with the ``--threads`` argument, e.g. `pyfit sample --threads 4 mymodel.hdf5 1000`.  You can both initialize and sample from the model with the `run` subcommand.  A reminder of all of these subcommands can be found by calling help, e.g., `pyfit.py -h`.
