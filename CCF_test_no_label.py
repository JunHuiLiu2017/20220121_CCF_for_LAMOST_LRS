import joblib
import numpy as np
from laspec.ccf import wxcorr_rvgrid
import time
import os
from laspec import mrs


def read_spectra_multi(fp_list, dir_path, wave):
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
    # mask_list = np.zeros((spec_num, len(wave)), dtype=float)
    flux_norm_array = np.zeros((spec_num, len(wave)), dtype=float)
    flux_norm_err_array = np.zeros((spec_num, len(wave)), dtype=float)
    for _ in range(spec_num):
        # mask_list[_] = np.interp(wave, spec_list[_].wave, spec_list[_].mask)
        flux_norm_array[_], flux_norm_err_array[_] = spec_list[_].interp_then_norm(wave)
    flux_norm_err_array[np.isnan(flux_norm_err_array)] = 10000
    flux_norm_array[np.isnan(flux_norm_array)] = 0
    return flux_norm_array, flux_norm_err_array, spec_list[0].ra, spec_list[0].dec

def do_CCF_fits_name(ccf_specs, work_path, fits_name, wave=np.arange(3950, 5750, 1)):
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
        rvgrid, ccf = wxcorr_rvgrid(wave, test_spec, CCF_wave, i, rv_grid=np.linspace(-500, 500, 50)) # MRS 40-50 rvgrid 10, #*** LRS rvgrid 30km/s ***#
        maxvalues.append(max(ccf))
        rvs.append(rvgrid[np.argmax(ccf)])

    #### 得到粗略参数后，做速度最优化，使得速度不是格点值。

    para_CCF[0] = int(obsid) #group id  RA DEC
    para_CCF[1:5] = CCF_specs['p_regli_CCF'][np.argmax(maxvalues)]
    para_CCF[5] = snr
    para_CCF[6] = rvs[np.argmax(maxvalues)]
    # plt.plot(CCF_wave, CCF_specs['flux_norm_regli_CCF'][np.argmax(maxvalues)], 'r-')
    # plt.plot(wave, test_spec, 'b-')
    # plt.show()
    return {'para_CCF': para_CCF}


def do_CCF_fits_path(ccf_specs, work_path, fits_path, wave=np.arange(3950, 5750, 1)):
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


def getFileName2(path, suffix):
    input_template_All = []
    input_template_All_Path = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if os.path.splitext(name)[-1] == suffix:
                input_template_All.append(name)
                input_template_All_Path.append(os.path.join(root, name))

    return input_template_All, input_template_All_Path


if __name__ == '__main__':
    wave = np.arange(3950, 5750, 1)
    CCF_wave = np.arange(3800, 6000, 1)

    CCF_specs = joblib.load(
        '/Users/liujunhui/PycharmProjects/20220121_CCF_for_LAMOST_LRS/2022_01_23_23_47_23_imitated_CCF_wl_3800_6000_300_.dump')

    work_path = '/Users/liujunhui/Desktop/2021workMac/202111012totallynew/test_data_group/Test_data_files/'
    input_template_All, input_template_All_Path = getFileName2(work_path, '.gz')

#######do_CCF_fits_path
    # start = time.time()
    # results = do_CCF_fits_path(ccf_specs=CCF_specs, work_path=work_path, fits_path=input_template_All[:3],
    #                  wave=np.arange(3950, 5750, 1))
    # print(results['para_CCF'])
    # end = time.time()
    # print(end - start)

#######do_CCF_fits_name
    start = time.time()
    for i in range(len(input_template_All)):
        results = do_CCF_fits_name(ccf_specs=CCF_specs, work_path=work_path, fits_name=input_template_All[i],
                        wave=np.arange(3950, 5750, 1))
        print(results['para_CCF'])
        end = time.time()
        print(end - start)


