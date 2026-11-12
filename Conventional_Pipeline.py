#!/usr/bin/env python
import argparse
import numpy as np
import time
import warnings
from tqdm import tqdm

def main():
    # Do the work
    t_start = time.time()
    if args.verbose:
        print('Run NODDI')
    noddi = NODDI(datafile=args.data,
                  maskfile=args.mask,
                  bvalsfile=args.bvals,
                  bvecsfile=args.bvecs)
    t_fit_start = time.time()

    maps = noddi.fit()

    fit_seconds = time.time() - t_fit_start

    if args.verbose:
        print(f'...fitting took {fit_seconds:.2f} seconds.')

    with open(args.output + "_fit_runtime.txt", "w") as f:
        f.write(f"fit_seconds: {fit_seconds:.2f}\n")
        f.write(f"fit_minutes: {fit_seconds / 60:.2f}\n")
        f.write(f"fit_hours: {fit_seconds / 3600:.2f}\n")

    # Save the results
    if args.verbose:
        print('Save maps')
    for m in maps:
        if args.verbose:
            print(f'... {m}')
        noddi.save_map(maps[m], args.output+'_'+m)
        # saving maps

    return 0


warnings.filterwarnings("ignore")
def noddi_vals(X,bvecs,bvals):
    # note the change of units to SI
    acq_scheme    = acquisition_scheme_from_bvalues(bvals*1E6, bvecs / np.linalg.norm(bvecs,axis=1,keepdims=True))
    stick         = cylinder_models.C1Stick()
    zeppelin      = gaussian_models.G2Zeppelin()
    ball          = gaussian_models.G1Ball()
    watson_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])
    watson_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'G2Zeppelin_1_lambda_par', 'partial_volume_0')
    watson_bundle.set_equal_parameter('C1Stick_1_lambda_par', 'G2Zeppelin_1_lambda_par')
    watson_bundle.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)

    mcmodel = MultiCompartmentModel(models=[ball,watson_bundle])
    mcmodel.set_fixed_parameter('G1Ball_1_lambda_iso',3.0e-9)

    print(mcmodel.parameter_names)
    def normalize_signal_each_row(data, bvals):
        b0_indices = np.where(bvals == 0)[0]
        if len(b0_indices) == 0:
            raise ValueError("No b0 images found in the provided b-values.")
        b0_mean_signal = np.mean(data[:, b0_indices], axis=1, keepdims=True)
        normalized_data = data / b0_mean_signal

        return normalized_data

    X_normalized = normalize_signal_each_row(X.copy(), bvals)

    chunk_size = 5000

    odi = np.zeros(X_normalized.shape[0])
    icvf = np.zeros(X_normalized.shape[0])
    fiso = np.zeros(X_normalized.shape[0])

    for start in tqdm(range(0, X_normalized.shape[0], chunk_size), desc="Conventional full brain"):
        end = min(start + chunk_size, X_normalized.shape[0])
        res = mcmodel.fit(acq_scheme, X_normalized[start:end])
        odi[start:end] = res.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']
        icvf[start:end] = res.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']
        fiso[start:end] = res.fitted_parameters['partial_volume_0']

    return {'OD_no_noise':odi, 'ICVF_no_noise':icvf, 'ISOVF_no_noise':fiso}

class NODDI(object):

    """Main NODDI class - uses DMIPY for model definition and fitting
    """
    def __init__(self, datafile, maskfile, bvalsfile, bvecsfile):
        """Constructor
        datafile, maskfile, bvalsfile, bvecsfile alls strings
        bvals MUST be in ms/mm^2 (e.g. b=1000, 2000 etc.)
        """
        self._data   = Image(datafile).data
        self._mask   = Image(maskfile).data
        self._bvecs  = np.loadtxt(bvecsfile)
        self._bvals  = np.loadtxt(bvalsfile)
        self._header = Image(maskfile).header
        self._bvecs  = self.check_bvecs(self._bvecs, self._bvals)


    @staticmethod
    def check_bvecs(bvecs, bvals):
        """DMIPY can't handle un-normalised bvecs or bvecs that are exactly zero.
        """
        if bvecs.shape[0]<bvecs.shape[1]:
            bvecs = bvecs.T
        bvecs[bvals<10,:] = [1,0,0]
        bvecs = bvecs / np.linalg.norm(bvecs,axis=1,keepdims=True)
        return bvecs

    def data_to_X(self, data=None):
        if data is None:
            data = self._data
        return np.reshape(data,(-1,data.shape[-1]))[self._mask.flatten()>0,:]

    def X_to_data(self, X):
        if X.ndim < 2:
            X = X[:,None]
            tmp = np.zeros((self._mask.size, X.shape[1]))
            tmp[self._mask.flatten()>0,:] = X
        return np.reshape(tmp, self._mask.shape + (X.shape[1],))

    def save_map(self, Map, filename):
        img = Image(self.X_to_data(Map),header=self._header)
        img.save(filename)

    def fit(self, data=None):
        if data is None:
            data = self._data
        X = self.data_to_X(data)
        return noddi_vals(X, self._bvecs, self._bvals)

def parse_cmdline():
    """Parse command line arguments
    Uses the same convention as FSL-DTIFIT
    """
    p = argparse.ArgumentParser("noddi_fit")
    p.add_argument('-k,--data',required=True,type=str,dest='data',
                   metavar='<IMAGE>',help='input DWI data')
    p.add_argument('-m,--mask',required=True,type=str,dest='mask',
                   metavar='<IMAGE>',help='brain mask')
    p.add_argument('-b,--bvals',required=True,type=str,dest='bvals',
                   metavar='<FILE>',help='bvals ASCII text file')
    p.add_argument('-r,--bvecs',required=False,type=str,dest='bvecs',
                   metavar='<FILE>',help='bvecs ASCII text file')
    p.add_argument('-o,--out',required=True,type=str,dest='output',
                   metavar='<IMAGE>',help='output image file')
    p.add_argument('-v,--verbose',help='switch on diagnostic messages',action="store_true",dest='verbose')

    return p.parse_args()



if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_cmdline()

    # Imports after parsing to speed up help
    from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
    from dmipy.signal_models           import gaussian_models, cylinder_models
    from dmipy.distributions           import distribute_models
    from dmipy.core.modeling_framework import MultiCompartmentModel
    from fsl.data.image import Image

    main()
