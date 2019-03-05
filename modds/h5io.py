"""
h5io.py

Convenience functions/classes for working with hdf5 files and saving the outputs
from MCMC sampling.

The HDF5Backend class manages an hdf5 file which stores the setup of a model 
along with the outputs of sampling from that model.

HDF5Backend instances can be created from existing hdf5 files, by providing a 
MeasurementModel instance and dictionary of settings, or through the parse_yaml
function, which will attempt to construct a model from a properly formatted yaml
file.
"""

import os
import warnings
import pkg_resources
from collections import OrderedDict

import numpy as np
import h5py
import emcee
import dill
from tqdm import tqdm
import ruamel.yaml as yaml

from colossus.cosmology.cosmology import setCosmology
from colossus.halo import (profile_dk14, profile_outer, profile_nfw,
                           profile_einasto)

from modds.parameter import Parameter
from modds.measurement import MeasurementModel
from modds import joint_priors

__all__ = ['HDF5Backend', 'parse_yaml']


def get_version(package):
    return pkg_resources.get_distribution(package).version


def visit(hdf5_file):
    """Recursively visit and print all hdf5 groups.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    """
    with h5py.File(hdf5_file, "r") as f:
        f.visititems(lambda key, value: print(key, value))


def read_dataset(hdf5_file, path):
    """Return a stored dataset at the specified path.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    path : str
        hdf5 style pathname

    Returns
    -------
    dataset : any
        The dataset stored at path
    """
    with h5py.File(hdf5_file, "r") as f:
        return f[path].value


def write_dataset(hdf5_file, path, data, overwrite=False):
    """Store a dataset at the specified path.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    path : str
        hdf5 style pathname
    data : array_like
    overwrite : bool, optional
        if True, allow overwriting of data, defaults to False
    """
    with h5py.File(hdf5_file, mode="a") as f:
        if (path in f) and (not overwrite):
            raise RuntimeError("Path {} exists, overwrite?".format(path))
        elif overwrite:
            del f[path]
        f[path] = data
        f.flush()


def initialize_dataset(hdf5_file, path, shape, maxshape=None,
                       compression="gzip", overwrite=False):
    """Initialize an array dataset at the specified path.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    path : str
        hdf5 style pathname
    shape : tuple
        initial dimensions of array
    maxshape : tuple, optional
        Defaults to None, in which case maxshape is set to shape.  If a 
        dimension is None, then the maxsize in that dimension is infinite.
    overwrite : bool, optional
        if True, allow overwriting of data, defaults to False
    """
    if maxshape is None:
        maxshape = shape
    with h5py.File(hdf5_file, mode="a") as f:
        if (path in f) and (not overwrite):
            raise RuntimeError("Path {} exists, overwrite?".format(path))
        elif overwrite:
            del f[path]
        f.create_dataset(path, shape, maxshape=maxshape,
                         compression=compression)
        f.flush()


def read_object(hdf5_file, path):
    """Return a stored generic python object at the specified path.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    path : str
        hdf5 style pathname

    Returns
    -------
    object
    """
    with h5py.File(hdf5_file, "r") as f:
        return dill.loads(f[path].value.tostring())


def write_object(hdf5_file, path, obj, overwrite=False):
    """Store a generic python object at the specified path.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    path : str
        hdf5 style pathname
    obj : object
    overwrite : bool, optional
        if True, allow overwriting of data, defaults to False
    """
    with h5py.File(hdf5_file, mode="a") as f:
        if (path in f) and (not overwrite):
            raise RuntimeError("Path {} exists, overwrite?".format(path))
        elif overwrite:
            del f[path]
        bytestring = dill.dumps(obj)
        f[path] = np.void(bytestring)
        f.flush()


def read_group(hdf5_file, path):
    """Return a group at the specified path as a dictionary.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    path : str
        hdf5 style pathname

    Returns
    -------
    group : dict
        Dictionary located at path
    """
    group = {}
    with h5py.File(hdf5_file, "r") as f:
        for key, value in f[path].items():
            if isinstance(value, h5py.Dataset):
                group[key] = value.value
            elif isinstance(value, h5py.Group):
                group[key] = read_group(hdf5_file, "/".join([path, key]))
            else:
                raise ValueError("Unknown type: " + str(value))
        return group


def write_group(hdf5_file, path, group, overwrite=False):
    """Write the group dictionary to the path on the hdf5 file.

    Keys of the dictionary `group` will be new paths in the hdf5 file.
    This will recursively write to the file for nested dictionaries.

    Parameters
    ----------
    hdf5_file : str
        hdf5 filename
    path : str
        hdf5 style pathname
    group : dict
        dictionary to store at path
    overwrite : bool, optional
        if True, allow overwriting of data, defaults to False
    """
    with h5py.File(hdf5_file, mode="a") as f:
        for key, value in group.items():
            new_path = "/".join([path, key])
            if (new_path in f) and (not overwrite):
                raise RuntimeError("Path {} exists, overwrite?".format(path))
            elif overwrite:
                del f[new_path]
            if isinstance(value, dict):
                # recurses!
                write_group(hdf5_file, new_path, value, overwrite=overwrite)
            else:
                f[new_path] = value
        f.flush()


class HDF5Backend():
    """Manages an hdf5 file and defines the structure of saved MCMC runs.

    Calling the constructor with a filename will load the filename and make
    convenience functions available to operate on the file, including sampling.

    Calling the constructor with a filename along with a model and a dictionary
    of settings will allow you to create a new hdf5 file in addition to loading
    it as an HDF5Backend instance.

    The actual structure of the hdf5 file should not be necessary to know to
    interact with the instance, but should you wish to manually interact with
    it, is specified as follows:

    /model : bytestring representation of MeasurementModel instance
    /version : str, version of colossus associated with the file
    /ndim : int, number of dimensions of parameter space
    /observables/R : radial locations of observed quantities
    /observables/q : observed quantity
    /observables/q_err : observational uncertainty

    /state/chain : (nwalker, niter, ndim)-shaped array of posterior samples
    /state/niter : int, number of iterations
    /state/nacc : (nwalker,)-shaped array of acceptance counts for walkers
    /state/ppc : (nwalker, niter, ngrid)-shaped array of posterior predictive 
                 checks
    /state/lnp : (nwalker, niter)-shaped array of log posterior values
    /state/lnl : (nwalker, niter)-shaped array of log likelihood values

    /settings/cosmo : Colossus cosmology (str, e.g., "planck15")
    /settings/quantity : one of {'density', 'enclosedMass', 'surfaceDensity',
                                 'deltaSigma'}
    /settings/r_grid : grid of radii for posterior predictive checks
    /settings/nwalkers : number of walkers in the ensemble

    Parameters
    ----------
    filename : str
    model : MeasurementModel instance
    settings : dict
        Keys should include cosmo, nwalkers, and r_grid.
    """

    def __init__(self, filename=None, model=None, settings=None,
                 overwrite=False):
        self.filename = filename
        if (model is None) and (settings is None):
            # load from file
            self.load(filename)
        else:
            assert (model is not None) and (settings is not None)
            # write to and load file
            self.dump(filename, model, settings, overwrite=overwrite)
            self.load(filename)

    def load(self, filename):
        """Sets the filename, reads the model, and sets the cosmology."""
        assert os.path.isfile(filename)
        self.filename = filename
        colossus_version = get_version("colossus")
        if colossus_version != self.version:
            warnings.warn("Current version of colossus ({}) does not match that"
                          " of the file ({})!".format(colossus_version, self.version))
        self._model = read_object(filename, "model")
        self._observables = self.model.observables
        self._quantity = self.model.quantity
        cosmo_name = read_dataset(filename, "/settings/cosmo")
        self._cosmo = setCosmology(cosmo_name)

    def dump(self, filename, model, settings, overwrite=False):
        """Stores the model/settings and initializes arrays."""
        if os.path.isfile(filename) and not overwrite:
            raise ValueError(
                "{} exists, do you want to overwrite it?".format(filename))
        f = filename
        write_object(f, "/model", model, overwrite=overwrite)
        write_dataset(f, "/version", get_version("colossus"),
                      overwrite=overwrite)
        write_dataset(f, "/ndim", len(model.parameters), overwrite=overwrite)
        write_group(f, "/observables", model.observables, overwrite=overwrite)
        write_group(f, "/settings", settings, overwrite=overwrite)
        write_dataset(f, "/settings/quantity", model.quantity,
                      overwrite=overwrite)
        ndim = self.ndim
        nwalkers = self.nwalkers
        ngrid = len(self.r_grid)
        initialize_dataset(f, "/state/chain", shape=(nwalkers, 0, ndim),
                           maxshape=(nwalkers, None, ndim), overwrite=overwrite)
        initialize_dataset(f, "/state/lnp", shape=(nwalkers, 0),
                           maxshape=(nwalkers, None), overwrite=overwrite)
        initialize_dataset(f, "/state/lnl", shape=(nwalkers, 0),
                           maxshape=(nwalkers, None), overwrite=overwrite)
        initialize_dataset(f, "/state/Mvir", shape=(nwalkers, 0),
                           maxshape=(nwalkers, None), overwrite=overwrite)
        initialize_dataset(f, "/state/cvir", shape=(nwalkers, 0),
                           maxshape=(nwalkers, None), overwrite=overwrite)
        initialize_dataset(f, "/state/rsp", shape=(nwalkers, 0),
                           maxshape=(nwalkers, None), overwrite=overwrite)
        initialize_dataset(f, "/state/gamma_min", shape=(nwalkers, 0),
                           maxshape=(nwalkers, None), overwrite=overwrite)
        initialize_dataset(f, "/state/ppc", shape=(nwalkers, 0, ngrid),
                           maxshape=(nwalkers, None, ngrid),
                           overwrite=overwrite)
        write_dataset(f, "/state/niter", 0, overwrite=overwrite)
        write_dataset(f, "/state/nacc", np.zeros(nwalkers),
                      overwrite=overwrite)

    def append_state(self, pos, lnp=None, ppc=None, lnl=None, Mvir=None,
                     cvir=None, rsp=None, gamma_min=None, acceptances=None):
        """Append sampling results to the existing datasets.

        Parameters
        ----------
        pos : array_like
            shape of (nwalkers, ndim) of parameter positions
        lnp : array_like
            shape of (nwalkers,) of log posterior values
        ppc : array_like
            shape of (nwalkers, ngrid) of posterior predictive checks
        lnl : array_like
            shape of (nwalkers,) of log likelihood values
        Mvir : array_like
            shape of (nwalkers,) of virial mass values
        cvir : array_like
            shape of (nwalkers,) of concentration values
        rsp : array_like
            shape of (nwalkers,) of splashback radius values
        gamma_min : array_like
            shape of (nwalkers,) of minimum gamma values

        acceptances : array_like
            shape of (nwalkers,) of acceptance counts (1 if accepted, 0 if not)
        """
        niter = self.niter
        with h5py.File(self.filename, mode="a") as f:
            # append to walker chains
            chain = f["/state/chain"]
            chain.resize((chain.shape[0], chain.shape[1] + 1, chain.shape[2]))
            chain[:, -1, :] = pos
            f["/state/niter"][...] = niter + 1
            # add to acceptance counts
            if acceptances is not None:
                old_nacc = f["/state/nacc"].value
                f["/state/nacc"][...] = old_nacc + acceptances
            # append to log posterior values
            if lnp is not None:
                x = f["/state/lnp"]
                x.resize((x.shape[0], x.shape[1] + 1))
                x[:, -1] = lnp
            # append to posterior predictive checks
            if ppc is not None:
                x = f["/state/ppc"]
                x.resize((x.shape[0], x.shape[1] + 1, x.shape[2]))
                x[:, -1, :] = ppc
            # append to log likelihood values
            if lnl is not None:
                x = f["/state/lnl"]
                x.resize((x.shape[0], x.shape[1] + 1))
                x[:, -1] = lnl
            if Mvir is not None:
                x = f["/state/Mvir"]
                x.resize((x.shape[0], x.shape[1] + 1))
                x[:, -1] = Mvir
            if cvir is not None:
                x = f["/state/cvir"]
                x.resize((x.shape[0], x.shape[1] + 1))
                x[:, -1] = cvir
            if rsp is not None:
                x = f["/state/rsp"]
                x.resize((x.shape[0], x.shape[1] + 1))
                x[:, -1] = rsp
            if gamma_min is not None:
                x = f["/state/gamma_min"]
                x.resize((x.shape[0], x.shape[1] + 1))
                x[:, -1] = gamma_min
            f.flush()

            
    def sample(self, niter, pool=None):
        """Sample from the model for niter iterations, optionally with a user
        provided pool of workers.
        """
        nwalkers = self.nwalkers
        ndim = self.ndim
        r_grid = self.r_grid
        try:
            mdef = read_dataset(self.filename, "/settings/mdef")
        except:
            mdef = "vir"
        # save all the things!
        kwargs = dict(return_lnlike=True,
                      return_profile=True,
                      return_vir=True,
                      return_sp=True,
                      r_grid=r_grid,
                      mdef=mdef)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.model, pool=pool,
                                        kwargs=kwargs)
        old_nacc = np.zeros(nwalkers)
        # if initial run, start by sampling from prior
        if self.niter == 0:
            pos = np.empty((nwalkers, ndim))
            for i, (name, param) in enumerate(self.model.parameters.items()):
                pos[:, i] = param.sample(nwalkers)
        # otherwise start from the last position
        else:
            pos = self.chain[:, -1, :]
        with tqdm(total=niter) as pbar:
            for result in sampler.sample(
                    pos, iterations=niter, storechain=False):
                new_pos = result[0]
                new_lnp = result[1]
                rstate = result[2]
                blobs = result[3]
                new_lnl = np.array([blobs[i][0] for i in range(nwalkers)])
                lens = [len(blobs[i][1]) for i in range(nwalkers)]
                new_q_grid = np.vstack([blobs[i][1] for i in range(nwalkers)])
                new_Mvir = np.array([blobs[i][2] for i in range(nwalkers)])
                new_cvir = np.array([blobs[i][3] for i in range(nwalkers)])
                new_rsp = np.array([blobs[i][4] for i in range(nwalkers)])
                new_gamma_min = np.array([blobs[i][5] for i in range(nwalkers)])
                delta_nacc = sampler.naccepted - old_nacc
                # need to save as copy, default is a reference!
                old_nacc = sampler.naccepted.copy()
                self.append_state(new_pos, lnp=new_lnp, ppc=new_q_grid,
                                  lnl=new_lnl, Mvir=new_Mvir, cvir=new_cvir,
                                  rsp=new_rsp, gamma_min=new_gamma_min,
                                  acceptances=delta_nacc)
                pbar.update()

    @property
    def model(self):
        return self._model

    @property
    def observables(self):
        return self._observables

    @property
    def quantity(self):
        return self._quantity

    @property
    def version(self):
        return read_dataset(self.filename, "/version")

    @property
    def ndim(self):
        return read_dataset(self.filename, "/ndim")

    @property
    def shape(self):
        return self.nwalkers, self.niter, self.ndim

    @property
    def settings(self):
        return read_group(self.filename, "/settings")

    @property
    def r_grid(self):
        return read_dataset(self.filename, "/settings/r_grid")

    @property
    def cosmo(self):
        return self._cosmo

    @property
    def nwalkers(self):
        return read_dataset(self.filename, "/settings/nwalkers")

    @property
    def chain(self):
        return read_dataset(self.filename, "/state/chain")

    @property
    def niter(self):
        return read_dataset(self.filename, "/state/niter")

    @property
    def nacc(self):
        return read_dataset(self.filename, "/state/nacc")

    @property
    def ppc(self):
        return read_dataset(self.filename, "/state/ppc")

    @property
    def acceptance_fraction(self):
        return self.nacc / self.niter

    @property
    def lnl(self):
        return read_dataset(self.filename, "/state/lnl")

    @property
    def lnp(self):
        return read_dataset(self.filename, "/state/lnp")


def parse_yaml(filename, overwrite=False):
    """Read in a yaml file and construct an hdf5 output file."""
    with open(filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

        # construct parameter objects
        parameters = []
        for pdict in config['model']['parameters']:
            name = list(pdict.keys())[0]
            prior = pdict[name]['prior']
            try:
                transform = pdict[name]['transform']
            except KeyError:
                transform = None
            parameters.append(Parameter(name, prior, transform))

        # construct observables
        if os.path.isfile(config['model']['observables']):
            data_filename = config['model']['observables']
            r, q, q_err = np.loadtxt(data_filename, delimiter=',').T
        else:
            r = config['model']['observables']['r']
            q = config['model']['observables']['q']
            q_err = config['model']['observables']['q_err']
        observables = OrderedDict([('r', r), ('q', q), ('q_err', q_err)])

        # construct constants dictionary
        try:
            constants = config['model']['constants']
        except KeyError:
            constants = None
        
        # construct profile
        profile_name = config['model']['profile']
        if 'z' in constants:
            z = constants['z']
        else:
            names = [p.name for p in parameters]
            idx = names.index('z')
            pz = parameters[idx]
            if pz.transform is not None:
                z = pz.inverse_transform(pz.prior.mean())
            else:
                z = pz.prior.mean()
            
        cosmo_name = config['settings']['cosmo']
        cosmo = setCosmology(cosmo_name)

        try:
            outer_term_name = config['model']['outer_term']
        except KeyError:
            outer_term_name = None
        if outer_term_name is not None:
            if profile_name.lower() == 'dk14':
                # halo mass parameters are arbitrary and will be changed,
                # *except* for the redshift!
                kwargs = dict(outer_term_names=[outer_term_name],
                              M=1e12, c=10, z=z, mdef='vir')
                profile = profile_dk14.getDK14ProfileWithOuterTerms(**kwargs)
            else:
                raise NotImplementedError(
                    "Only supports outer terms with DK14.")
        else:
            profile_classes = dict('nfw', profile_nfw.NFWProfile,
                                   'einasto', profile_einasto.EinastoProfile,
                                   'dk14', profile_dk14.DK14Profile)
            kwargs = dict(M=1e12, c=10, z=z, mdef='vir')
            profile = profile_classes[profile_name](**kwargs)

        quantity = config['model']['quantity']

        # construct any joint priors
        if 'priors' in config['model']:
            priors = []
            for prior in config['model']['priors']:
                try:
                    priors.append(getattr(joint_priors, prior))
                except:
                    raise ValueError('{} not found as a valid joint '
                                     'prior'.format(prior))
        else:
            priors = None
                    
        # check for r_grid
        settings = config['settings']
        if isinstance(settings['r_grid'], int):
            ngrid = settings['r_grid']
            rmin = np.amin(observables['r'])
            rmax = np.amax(observables['r'])
            settings['r_grid'] = np.logspace(np.log10(rmin), np.log10(rmax),
                                             ngrid)

        model = MeasurementModel(profile=profile, observables=observables,
                                 parameters=parameters, quantity=quantity,
                                 constants=constants, priors=priors)

        if 'output' not in config:
            outfilename = filename.split('.yaml')[0] + '.hdf5'
        else:
            outfilename = config['output']
        outfile = HDF5Backend(outfilename, model=model, settings=settings,
                              overwrite=overwrite)
        return outfile
