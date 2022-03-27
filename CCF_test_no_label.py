import joblib
import numpy as np
from laspec.ccf import wxcorr_rvgrid
import time
import os
from laspec import mrs
import pandas as pd
from laspec.ccf import RVM
import matplotlib.pyplot as plt
from astropy import constants
from laspec.normalization import normalize_spectrum_general

SOL_kms = constants.c.value / 1000


def wave_rv_my(wave_range, rv=0):
    """ calculate RV-corrected wavelength array

    Parameters
    ----------
    rv: float
        radial velocity in km/s

    Args:
        wave_range: the wave range

    """
    return wave_range / (1 + rv / SOL_kms)


def interp_my(new_wave, old_wave, flux, rv=0):
    """ interpolate to a new wavelength grid """
    return np.interp(new_wave, wave_rv_my(old_wave, rv), flux)


def positive_interp_then_norm(mrs_file, wave_range, rv=0):
    """
    By this process, the negative flux values are neglected,
    then their original flux values are interpolated as positive values.
    Meanwhile, for the negative norm flux they are set as 1.

    Args:
        mrs_file: the mrs file made already
        wave_range: the work wave, wave=np.arange(3950, 5750, 1)
        rv: the rv of spectra, should be set as 0.
        Because we measure the rv, the rv values built into the Mrs File from LAMOST.

    Returns:


    """
    flux_interp = mrs_file.interp(wave_range, rv=rv)
    my_mask = flux_interp > 0
    masked_flux_interp = flux_interp[my_mask]
    masked_wave = wave[my_mask]
    positive_flux_interp = interp_my(wave, masked_wave, masked_flux_interp, rv=rv)
    flux_norm_my, flux_cont_my = normalize_spectrum_general(wave_range, positive_flux_interp)
    flux_norm_err_my = np.interp(wave, wave_rv_my(mrs_file.wave, rv), mrs_file.flux_err) / flux_cont_my
    return flux_norm_my, flux_norm_err_my


def read_spectra_multi(fp_list, dir_path, wave=np.arange(3950, 5750, 1)):
    """Read a small spectral list of one object, \
    this list holds all spectra of this object observed in different epochs.

    Parameters
    ----------
    fp_list:
        list of file paths
    wave:
        interpolation grid

    Returns
    -------
    normalized flux and the corresponding error

    Notes: First interpolation, then norm

    """
    spec_list = [mrs.MrsSpec.from_lrs(dir_path + fp.split('/')[-1]) for fp in fp_list]
    spec_num = len(spec_list)
    mask_list = np.zeros((spec_num, len(wave)), dtype=float)
    ivar_list = np.zeros((spec_num, len(wave)), dtype=float)
    flux_norm_array = np.zeros((spec_num, len(wave)), dtype=float)
    flux_norm_err_array = np.zeros((spec_num, len(wave)), dtype=float)
    snr_array = np.zeros((spec_num, 1), dtype=float)
    obsid_array = np.zeros((spec_num, 1), dtype=float)
    for _ in range(spec_num):
        mask_list[_] = np.interp(wave, spec_list[_].wave, spec_list[_].mask)
        ivar_list[_] = np.interp(wave, spec_list[_].wave, spec_list[_].ivar)
        flux_norm_array[_], flux_norm_err_array[_] = positive_interp_then_norm(spec_list[_], wave)
        snr_array[_] = spec_list[_].snr
        obsid_array[_] = spec_list[_].obsid
    flux_norm_err_array[np.isnan(flux_norm_err_array)] = 10000
    flux_norm_array[np.isnan(flux_norm_array)] = 0

    # plt.plot(wave, mask_list[0], 'k-')
    # plt.plot(wave, mask_list[1], 'r-')
    # plt.plot(spec_list[0].wave, spec_list[0].flux, 'k-')
    # plt.plot(spec_list[1].wave, spec_list[1].flux, 'k-')
    # plt.plot(wave, ivar_list[0], 'k-')
    # plt.plot(wave, ivar_list[1], 'r-')
    # plt.show()

    ###filter the abnormal data###
    delete_arrays = []
    for _i, _j in enumerate(flux_norm_array):
        if len(_j[_j==0]) == len(wave):
            delete_arrays.append(_i)
        # elif len(np.unique(_j)) < 1000:
            # delete_arrays.append(_i)
    for _k, _m in enumerate(mask_list):
        if len(_m[_m!=0]) == len(wave):
            delete_arrays.append(_k)
    delete_tuple = tuple(np.unique(delete_arrays))
    flux_norm_array = np.delete(flux_norm_array, delete_tuple, axis=0)
    flux_norm_err_array = np.delete(flux_norm_err_array, delete_tuple, axis=0)
    snr_array = np.delete(snr_array, delete_tuple, axis=0)
    obsid_array = np.delete(obsid_array, delete_tuple, axis=0)
    ###filter the abnormal data###
    # print(flux_norm_array)
    return flux_norm_array, flux_norm_err_array, snr_array, obsid_array


def do_CCF_fits_name(ccf_specs, work_path, fits_name):
    """

    Args:
        ccf_specs: The template CCF spectra (with labels: teff, logg, M/H, alpha/M) to do CCF.
        ccf_params:
        ------work_path: The path of the file of observed spectra. ------delete
        fits_path: the fits name
        id : list for observed id (row number)

        wave: the range of wavelength

    Returns:
        dict  : the input id, CCF_maxs,
        para_CCF: obsid of LAMOST, teff, logg, M/H, alpha/M, snr, rv

    """
    para_CCF = np.zeros((7), dtype=float)
    fits = mrs.MrsSpec.from_lrs(work_path + fits_name)
    test_spec = fits.interp_then_norm(wave, rv=0)[0]
    snr = fits.snr
    obsid = fits.obsid
    maxvalues = []
    rvs = []
    for i in ccf_specs['flux_norm_regli_CCF']:
        rvgrid, ccf = wxcorr_rvgrid(wave, test_spec, CCF_wave, i, rv_grid=np.linspace(-500, 500,
                                                                                      50))  # MRS 40-50 rvgrid 10, #*** LRS rvgrid 30km/s ***#
        maxvalues.append(max(ccf))
        rvs.append(rvgrid[np.argmax(ccf)])

    #### 得到粗略参数后，做速度最优化，使得速度不是格点值。

    para_CCF[0] = int(obsid)  # group id  RA DEC
    para_CCF[1:5] = CCF_specs['p_regli_CCF'][np.argmax(maxvalues)]
    para_CCF[5] = snr
    para_CCF[6] = rvs[np.argmax(maxvalues)]
    # plt.plot(CCF_wave, CCF_specs['flux_norm_regli_CCF'][np.argmax(maxvalues)], 'r-')
    # plt.plot(wave, test_spec, 'b-')
    # plt.show()
    return {'para_CCF': para_CCF}


def do_CCF_fits_path(ccf_specs, work_path, fits_path):
    """

    Args:
        ccf_specs: The template CCF spectra (with labels: teff, logg, M/H, alpha/M) to do CCF.
        work_path: The path of the file of observed spectra.
        fits_path: the fits name
        test_size: the size of single CCF
        wave: the range of wavelength

    Returns:
        para_CCF: obsid of LAMOST, teff, logg, M/H, alpha/M, snr, rv

    """
    test_size = len(fits_path)
    para_CCF = np.zeros((test_size, 7), dtype=float)

    for j in range(test_size):
        if j % 1 == 0:
            print(j, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
        fits = mrs.MrsSpec.from_lrs(work_path + fits_path[j])
        test_spec = fits.interp_then_norm(wave, rv=0)[0]
        snr = fits.snr
        obsid = fits.obsid
        maxvalues = []
        rvs = []
        for i in ccf_specs['flux_norm_regli_CCF']:
            rvgrid, ccf = wxcorr_rvgrid(wave, test_spec, CCF_wave, i, rv_grid=np.linspace(-250, 250, 50))
            maxvalues.append(max(ccf))
            rvs.append(rvgrid[np.argmax(ccf)])

        para_CCF[j][0] = int(obsid)
        para_CCF[j][1:5] = CCF_specs['p_regli_CCF'][np.argmax(maxvalues)]
        para_CCF[j][5] = snr
        para_CCF[j][6] = rvs[np.argmax(maxvalues)]
        # plt.plot(CCF_wave, CCF_specs['flux_norm_regli_CCF'][np.argmax(maxvalues)], 'r-')
        # plt.plot(wave, test_spec, 'b-')
        # plt.show()
    return {'para_CCF': para_CCF}


def do_CCF_flux_list(ccf_specs, flux_norm_array, snr_array, obsid_array, CCF_wave=np.arange(3800, 6000, 1), wave=np.arange(3950, 5750, 1)):
    """
    Do the CCF for the flux list.
    Args:
        ccf_specs: the templated spectra of CCF
        flux_norm_array: the list of normalized flux of observed spectra
        snr_array: the snr of these observed spectra
        obsid_array: the obsid of these observed spectra

    Returns:
        the parameters list of CCF
        para_CCF: obsid of LAMOST, teff, logg, M/H, alpha/M, rv, snr

    """
    test_size = len(flux_norm_array)
    template_size = len(ccf_specs['flux_norm_regli_CCF'])
    para_CCF = np.zeros((test_size, 8), dtype=float)
    max_ccfs = np.zeros((template_size, test_size), dtype=float)
    rvs = np.zeros((template_size, test_size), dtype=float)

    # do ccf for each spectra and find the best one template for one source (with multiple spectra that listed in
    # spectra list).
    for _i in range(template_size):
        for _j in range(test_size):
            rvgrid, ccf = wxcorr_rvgrid(wave, flux_norm_array[_j], CCF_wave, ccf_specs['flux_norm_regli_CCF'][_i],
                                        rv_grid=np.linspace(-500, 500, 50))
            max_ccfs[_i][_j] = max(ccf)
            rvs[_i][_j] = rvgrid[np.argmax(ccf)]

    # Get the best template and its index, then get the parameters and flux of this template.
    max_ccfs_mean_index = np.argmax(max_ccfs.max(axis=1))
    best_template_paras = ccf_specs['p_regli_CCF'][max_ccfs_mean_index]
    best_template_flux = ccf_specs['flux_norm_regli_CCF'][max_ccfs_mean_index]

    # plt.plot(wave, flux_norm_array[0], 'k-')
    # plt.plot(CCF_wave, best_template_flux, 'g-')
    # plt.show()
    # Utilized the row velocities to get the precise values. By the way, get the return.
    rvm = RVM(best_template_paras, CCF_wave, best_template_flux)
    for _k in range(test_size):
        rvr = rvm.measure(wave_obs=wave, flux_obs=flux_norm_array[_k])
        para_CCF[_k][0] = obsid_array[_k][0]
        para_CCF[_k][1:5] = best_template_paras
        para_CCF[_k][5] = rvr['rv_opt']
        para_CCF[_k][6] = snr_array[_k][0]
        para_CCF[_k][7] = np.max(max_ccfs.max(axis=1))
    print(para_CCF)
    return para_CCF


def getFileName2(path, suffix):
    input_template_All = []
    input_template_All_Path = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if os.path.splitext(name)[-1] == suffix:
                input_template_All.append(name)
                input_template_All_Path.append(os.path.join(root, name))

    return input_template_All, input_template_All_Path


def _one_task(fits_list, ccf_specs):
    work_path = './20220313_CCF_test_grouped_fits/'
    flux_norm_array, flux_norm_err_array, snr_array, obsid_array = read_spectra_multi(fits_list, work_path)
    if len(flux_norm_array) != 0:
        para_CCF = do_CCF_flux_list(ccf_specs, flux_norm_array, snr_array, obsid_array)
    else:
        para_CCF = []
    return para_CCF


if __name__ == '__main__':
    # Load CCF wave range, template and its corresponding parameters.
    CCF_wave = np.arange(3800, 6000, 1)
    CCF_specs = joblib.load('./2022_03_16_14_08_52_imitated_CCF_wl_3800_6000_1000_.dump')

    # Load observed spectral wave range, the folder of fits file and its parameters recorded file.
    wave = np.arange(3950, 5750, 1)
    work_path = './20220313_CCF_test_grouped_fits/'
    data_csv_path = './20220313_snrg_30_Grpeddata_CCF_test.csv'
    data_csv = pd.read_csv(data_csv_path)

    # To group the observed spectra based on their "GroupID".
    Grouped_data_csv = data_csv.groupby(data_csv['GroupID'])
    Unique_GroupID = pd.unique(data_csv['GroupID'])  # get unique GroupID

    start = time.time()

#   multiprocess
    fits_lists = []
    for _i in Unique_GroupID:
        fits_list = pd.unique(Grouped_data_csv.get_group(_i)['combined_file'])
        fits_lists.append(fits_list)
    # print(fits_lists[-1])
    # _one_task(fits_lists[-1], CCF_specs)
    result = joblib.Parallel(n_jobs=1, backend="multiprocessing")(joblib.delayed(_one_task)(_, CCF_specs) for _ in fits_lists[:])
    #print(result)
    end = time.time()
    print(end - start)