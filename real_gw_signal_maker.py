from __future__ import division, print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

import bilby
from bilby.core.prior import Uniform
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters, generate_all_bbh_parameters

from gwpy.timeseries import TimeSeries

import os
from sys import exit

print(bilby.__version__)

time_of_event = 1126259462.4

H1 = bilby.gw.detector.get_empty_interferometer("H1")
L1 = bilby.gw.detector.get_empty_interferometer("L1")

# Definite times in relatation to the trigger time (time_of_event), duration and post_trigger_duration
post_trigger_duration = 0.5
duration = 1
analysis_start = time_of_event + post_trigger_duration - duration

# Use gwpy to fetch the open data
#H1_analysis_data = TimeSeries.fetch_open_data(
#    "H1", analysis_start, analysis_start + duration, sample_rate=1024, cache=True)

#L1_analysis_data = TimeSeries.fetch_open_data(
#    "L1", analysis_start, analysis_start + duration, sample_rate=1024, cache=True)

H1_analysis_data = TimeSeries.find(
    "H1:DCS-CALIB_STRAIN_C01", analysis_start, analysis_start + duration)
H1_analysis_data = TimeSeries.resample(H1_analysis_data, 1024)

L1_analysis_data = TimeSeries.find(
    "H1:DCS-CALIB_STRAIN_C01", analysis_start, analysis_start + duration)
L1_analysis_data = TimeSeries.resample(L1_analysis_data, 1024)

H1_analysis_data.plot()
#plt.savefig('/home/hunter.gabbard/public_html/random_stuff/test.png')

H1.set_strain_data_from_gwpy_timeseries(H1_analysis_data)
L1.set_strain_data_from_gwpy_timeseries(L1_analysis_data)


# use predefined psd file
# If user is specifying PSD files
psd_file = '/home/hunter.gabbard/.local/lib/python3.6/site-packages/bilby/gw/detector/noise_curves/aLIGO_early_asd.txt'
H1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=psd_file)
L1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=psd_file)

"""
psd_duration = duration * 32
psd_start_time = analysis_start - psd_duration

#H1_psd_data = TimeSeries.fetch_open_data(
#    "H1", psd_start_time, psd_start_time + psd_duration, sample_rate=4096, cache=True)

#L1_psd_data = TimeSeries.fetch_open_data(
#    "L1", psd_start_time, psd_start_time + psd_duration, sample_rate=4096, cache=True)
H1_psd_data = TimeSeries.find(
    "H1:GDS-CALIB_STRAIN", psd_start_time, psd_start_time + psd_duration)
H1_psd_data = TimeSeries.resample(H1_psd_data, 1024)

L1_psd_data = TimeSeries.find(
    "L1:GDS-CALIB_STRAIN", psd_start_time, psd_start_time + psd_duration)
L1_psd_data = TimeSeries.resample(L1_psd_data, 1024)

psd_alpha = 2 * H1.strain_data.roll_off / duration
H1_psd = H1_psd_data.psd(fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median")
L1_psd = L1_psd_data.psd(fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median")

H1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=H1_psd.frequencies.value, psd_array=H1_psd.value)
L1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=H1_psd.frequencies.value, psd_array=L1_psd.value)
"""

"""
fig, ax = plt.subplots()
idxs = H1.strain_data.frequency_mask  # This is a boolean mask of the frequencies which we'll use in the analysis
ax.loglog(H1.strain_data.frequency_array[idxs],
          np.abs(H1.strain_data.frequency_domain_strain[idxs]))
ax.loglog(H1.power_spectral_density.frequency_array[idxs],
          H1.power_spectral_density.asd_array[idxs])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Strain [strain/$\sqrt{Hz}$]")
"""

H1.maximum_frequency = 1024
L1.maximum_frequency = 1024

"""
fig, ax = plt.subplots()
idxs = H1.strain_data.frequency_mask
ax.loglog(H1.strain_data.frequency_array[idxs],
          np.abs(H1.strain_data.frequency_domain_strain[idxs]))
ax.loglog(H1.power_spectral_density.frequency_array[idxs],
          H1.power_spectral_density.asd_array[idxs])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Strain [strain/$\sqrt{Hz}$]")
plt.savefig('/home/hunter.gabbard/public_html/random_stuff/test.png')
"""

prior = bilby.core.prior.PriorDict()
#prior.pop('chirp_mass')
prior['mass_ratio'] = bilby.gw.prior.Constraint(minimum=0.125, maximum=1, name='mass_ratio', latex_label='$q$', unit=None)
prior['mass_1'] = Uniform(name='mass_1', minimum=5, maximum=100,unit='$M_{\odot}$')
prior['mass_2'] = Uniform(name='mass_2', minimum=5, maximum=100,unit='$M_{\odot}$')
prior['phase'] = Uniform(name="phase", minimum=0, maximum=2*np.pi)
prior['geocent_time'] = Uniform(name="geocent_time", minimum=time_of_event-0.1, maximum=time_of_event+0.1)
prior['a_1'] =  0.0
prior['a_2'] =  0.0
prior['tilt_1'] =  0.0
prior['tilt_2'] =  0.0
prior['phi_12'] =  0.0
prior['phi_jl'] =  0.0
prior['dec'] =  -1.2232
prior['ra'] =  2.19432
prior['theta_jn'] =  1.89694
prior['psi'] =  0.532268
prior['luminosity_distance'] = Uniform(name='luminosity_distance', minimum=100, maximum=1000, unit='Mpc')

# First, put our "data" created above into a list of intererometers (the order is arbitrary)
interferometers = [H1, L1]

# Next create a dictionary of arguments which we pass into the LALSimulation waveform - we specify the waveform approximant here
waveform_arguments = dict(
    waveform_approximant='IMRPhenomPv2', reference_frequency=20., minimum_frequency=20.0)

# Next, create a waveform_generator object. This wraps up some of the jobs of converting between parameters etc
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments,
    parameter_conversion=convert_to_lal_binary_black_hole_parameters)

# Save whitened waveform
signal_fd_all = np.zeros((len(interferometers), 1024))
Nt = int(1024*duration)
for i in range(len(interferometers)):
    signal_fd = interferometers[i].strain_data.frequency_domain_strain
    signal_fd = signal_fd/interferometers[i].amplitude_spectral_density_array
    signal_fd = np.sqrt(2.0*Nt)*np.fft.irfft(signal_fd)
    signal_fd_all[i,:] = signal_fd

try:
    os.mkdir('short')
except:
    print('Directory already exists')
try:
    os.mkdir('short/test_waveforms/')
    os.mkdir('short/test_dynesty1/')
except:
    print('Directory already exists')

hf = h5py.File('short/test_waveforms/GW150914_0.h5py')
hf.create_dataset('noisy_waveforms', data=signal_fd_all)
hf.create_dataset('noisefree_waveforms', data=np.array([]))
hf.create_dataset('x_data', data=np.array([]))
hf.close()

# Finally, create our likelihood, passing in what is needed to get going
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers, waveform_generator, priors=prior,
    time_marginalization=False, phase_marginalization=True, distance_marginalization=False)

result = bilby.run_sampler(
    likelihood, prior, sampler='dynesty', outdir='short', label="GW150914",
#    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    save='hdf5', nlive=500, dlogz=3, plot=True  # <- Arguments are used to make things fast - not recommended for general use
)

# save test sample waveform
hf = h5py.File('short/test_dynesty1', 'w')
# loop over randomised params and save samples
for q,qi in result.posterior.items():
    name = q + '_post'
    print('saving PE samples for parameter {}'.format(q))
    hf.create_dataset(name, data=np.array(qi))
hf.close()

print('Generated posterior samples')
exit()

##################################################################################
result_short.posterior
result_short.posterior["chirp_mass"]
Mc = result_short.posterior["chirp_mass"].values
lower_bound = np.quantile(Mc, 0.05)
upper_bound = np.quantile(Mc, 0.95)
median = np.quantile(Mc, 0.5)
print("Mc = {} with a 90% C.I = {} -> {}".format(median, lower_bound, upper_bound))

fig, ax = plt.subplots()
ax.hist(result_short.posterior["chirp_mass"], bins=20)
ax.axvspan(lower_bound, upper_bound, color='C1', alpha=0.4)
ax.axvline(median, color='C1')
ax.set_xlabel("chirp mass")
plt.show()

result_short.plot_corner(parameters=["chirp_mass", "mass_ratio", "geocent_time", "phase"], prior=True)

parameters = dict(mass_1=36.2, mass_2=29.1)
result_short.plot_corner(parameters)

result_short.priors
result_short.sampler_kwargs["nlive"]
print("ln Bayes factor = {} +/- {}".format(
    result_short.log_bayes_factor, result_short.log_evidence_err))


