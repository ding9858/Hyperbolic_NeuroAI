#!/usr/bin/env python
import argparse
import torch
import numpy as np
import time
import warnings
from torch import pi
from sbi.analysis import pairplot
from sbi.utils import BoxUniform
from sbi.inference import SNPE_C, simulate_for_sbi
from torch.distributions import Uniform
from sbi.utils import MultipleIndependent
from tqdm import tqdm
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

def main():
    # Do the work
    t_start = time.time()
    if args.verbose:
        print('Run NODDI')
    noddi = NODDI(datafile=args.data,
                  maskfile=args.mask,
                  bvalsfile=args.bvals,
                  bvecsfile=args.bvecs)
    maps = noddi.fit()
    if args.verbose:
        print(f'...fitting took {time.time()-t_start} seconds.')

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

    def forward(p):
        theta,phi,SD1WatsonDistributed_1_SD1Watson_1_odi,SD1WatsonDistributed_1_partial_volume_0,partial_volume_0,partial_volume_1, noise = p
        parameters_dict = {
        'SD1WatsonDistributed_1_SD1Watson_1_mu': np.array([theta,phi], dtype=np.float64),
        'SD1WatsonDistributed_1_SD1Watson_1_odi': np.array([SD1WatsonDistributed_1_SD1Watson_1_odi], dtype=np.float64),
        'SD1WatsonDistributed_1_partial_volume_0': np.array([SD1WatsonDistributed_1_partial_volume_0], dtype=np.float64),
        'partial_volume_0': np.array([partial_volume_0], dtype=np.float64),
        'partial_volume_1': np.array([partial_volume_1], dtype=np.float64)
        }
        signal = mcmodel.simulate_signal(acq_scheme,parameters_dict)
        noise_std = np.array(noise, dtype=np.float64)
        noise_part = np.random.randn(*signal.shape) * noise_std
        return signal + noise_part


    Box_prior = BoxUniform(low  = torch.tensor([ 0.,  -pi, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]),
                     high =   torch.tensor([ pi,   pi, 0.9999, 0.9999, 0.9999, 0.9999, 0.1]))



    prior, num_parameters, prior_returns_numpy = process_prior(Box_prior)
    simulator = process_simulator(forward, prior, prior_returns_numpy)
    check_sbi_inputs(simulator, prior)
    theta = prior.sample((30000,))
    x = simulator(theta)

    theta_nan = torch.isnan(theta).any(dim=1)
    x_nan = torch.isnan(x).any(dim=1)
    nan_rows = theta_nan | x_nan

    theta_inf = torch.isinf(theta).any(dim=1)
    x_inf = torch.isinf(x).any(dim=1)
    inf_rows = theta_inf | x_inf

    bad_rows = nan_rows | inf_rows
    valid_rows = ~bad_rows

    theta_cleaned = theta[valid_rows]
    x_cleaned = x[valid_rows]

    print("Total simulations:", theta.shape[0])
    print("NaN simulations:", int(nan_rows.sum()))
    print("Inf simulations:", int(inf_rows.sum()))
    print("Invalid simulations:", int(bad_rows.sum()))
    print("Valid simulations:", theta_cleaned.shape[0])


    inference = SNPE_C(prior=prior)
    inference = inference.append_simulations(theta_cleaned, x_cleaned)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    N = X_normalized.shape[0]
    print(N)
    all_theta_values = np.zeros(N)
    all_phi_values = np.zeros(N)
    all_odi_values = np.zeros(N)
    all_watson_partial_volume_0_values = np.zeros(N)
    all_partial_volume_0_values = np.zeros(N)
    all_partial_volume_1_values = np.zeros(N)
    all_noise_values = np.zeros(N)

    all_failure_values = np.zeros(N)
    all_bad_signal_values = np.zeros(N)
    all_sampling_exception_values = np.zeros(N)
    all_bad_posterior_values = np.zeros(N)

    for i, element in tqdm(enumerate(X_normalized), total=N, desc="SBI full brain"):

        if not np.isfinite(element).all():
            all_failure_values[i] = 1
            all_bad_signal_values[i] = 1
            continue

        try:
            samples = posterior.sample((100,), x=element, show_progress_bars=False)

        except Exception:
            all_failure_values[i] = 1
            all_sampling_exception_values[i] = 1
            continue

        if not torch.isfinite(samples).all():
            all_failure_values[i] = 1
            all_bad_posterior_values[i] = 1
            continue

        mean_p = torch.mean(samples, dim=0)

        if not torch.isfinite(mean_p).all():
            all_failure_values[i] = 1
            all_bad_posterior_values[i] = 1
            continue

        theta,phi,odi,watson_p0,p0,p1,noise = [float(x) for x in mean_p]
        all_theta_values[i] = theta
        all_phi_values[i] = phi
        all_odi_values[i] = odi
        all_watson_partial_volume_0_values[i] = watson_p0
        all_partial_volume_0_values[i] = p0
        all_partial_volume_1_values[i] = p1
        all_noise_values[i] = noise

    print("Total voxels:", N)
    print("Successful voxels:", N - int(np.sum(all_failure_values)))
    print("Failed voxels:", int(np.sum(all_failure_values)))
    print("Bad observed signals:", int(np.sum(all_bad_signal_values)))
    print("Sampling exceptions:", int(np.sum(all_sampling_exception_values)))
    print("Non-finite posterior:", int(np.sum(all_bad_posterior_values)))

    np.savetxt(args.output + "_failed_voxel_indices.txt", np.where(all_failure_values == 1)[0], fmt="%d")

    return {'SBI_SNPE_C_ODI':all_odi_values,
            'SBI_SNPE_C_ICVF':all_watson_partial_volume_0_values,
            'SBI_SNPE_C_ISOVF':all_partial_volume_0_values,
            'SBI_SNPE_C_Noise':all_noise_values,
            'SBI_SNPE_C_P1':all_partial_volume_1_values,
            'SBI_failure':all_failure_values,
            'SBI_bad_signal':all_bad_signal_values,
            'SBI_sampling_exception':all_sampling_exception_values,
            'SBI_bad_posterior':all_bad_posterior_values}

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
