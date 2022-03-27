import joblib
import matplotlib
import numpy as np
from laspec.normalization import normalize_spectrum_general
from joblib import dump, load
from CCF_test_no_label import do_CCF_flux_list
from astropy import constants

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


def read_test_spec(mrs_file, true_para, wave=np.arange(3950, 5750, 1)):
    """Read a small spectral list of one object, \
    this list holds all spectra of this object observed in different epochs.
    *******COPY FROM read_spectra_multi in CCF_test_no_label.py

    By this process, the negative flux values are neglected,
    then their original flux values are interpolated as positive values.
    Meanwhile, for the negative norm flux they are set as 1.

    Parameters
    ----------
    mrs_file:
        the mrs file in test sample.
    true_para:
        the parameters of the test sample, including the teff, logg, M/H and alpha/M from APOGEE, and rv from LAMOST.

    Returns
    -------
    normalized flux and the corresponding error

    Notes: First interpolation, then norm


    """
    true_para_list = list(true_para)
    spec_list = [mrs_file]
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
    flux_norm_array[np.isnan(flux_norm_array)] = 1
    flux_norm_array[flux_norm_array < 0] = 1
    flux_norm_array[flux_norm_array > 3] = 1

    ###filter the abnormal data###
    delete_arrays = []
    for _i, _j in enumerate(flux_norm_array):
        if len(_j[_j == 0]) == len(wave):
            delete_arrays.append(_i)
        # elif len(np.unique(_j)) < 1000:
        # delete_arrays.append(_i)ªªªª
    for _k, _m in enumerate(mask_list):
        if len(_m[_m != 0]) == len(wave):
            delete_arrays.append(_k)
    delete_tuple = tuple(np.unique(delete_arrays))
    flux_norm_array = np.delete(flux_norm_array, delete_tuple, axis=0)
    flux_norm_err_array = np.delete(flux_norm_err_array, delete_tuple, axis=0)
    snr_array = np.delete(snr_array, delete_tuple, axis=0)
    obsid_array = np.delete(obsid_array, delete_tuple, axis=0)
    return flux_norm_array, flux_norm_err_array, snr_array, obsid_array, true_para_list


def _one_task(mrs_file, true_para, ccf_specs):
    flux_norm_array, flux_norm_err_array, snr_array, obsid_array, true_para_list = read_test_spec(mrs_file, true_para)
    para_CCF = do_CCF_flux_list(ccf_specs, flux_norm_array, snr_array, obsid_array)
    para_CCF = para_CCF.tolist()
    para = [para_CCF[0], true_para_list]
    return para


if __name__ == '__main__':
    wave = np.arange(3950, 5750, 1)
    CCF_wave = np.arange(3800, 6000, 1)

    CCF_specs = joblib.load(
        '/Users/liujunhui/PycharmProjects/20220121_CCF_for_LAMOST_LRS/2022_03_16_14_08_52_imitated_CCF_wl_3800_6000_1000_.dump')

    test_data = load('/Users/liujunhui/PycharmProjects/20220121_CCF_for_LAMOST_LRS/small_sample_3000.dump')
    params = test_data['params']
    spec_list = test_data['spec_list']

    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'  # 将刻度显示成朝里
    matplotlib.rcParams['ytick.right'] = 'True'
    matplotlib.rcParams['xtick.top'] = 'True'  # 设置x轴上方横线显示刻度
    matplotlib.rcParams['grid.linestyle'] = '--'  # 设置标度线的风格

    ###############################################################################################################
    # show the distribution of CCF spectra.
    # fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    # cm2 = plt.cm.get_cmap('jet')
    # im2 = ax1.scatter(CCF_specs['p_regli_CCF'][:, 0], CCF_specs['p_regli_CCF'][:, 1], s=30,
    #                   c=CCF_specs['p_regli_CCF'][:, 2], marker="o", edgecolor='k', alpha=.8, cmap=cm2, vmin=-2,
    #                   vmax=0.8)
    # position2 = fig1.add_axes([0.92, 0.12, 0.02, 0.76])  # 位置[左,下,右,上]
    # c2 = plt.colorbar(im2, cax=position2, orientation='vertical')  # 方向
    # c2.set_label("[Fe/H] [dex]", fontsize = 18)
    # ax1.grid(True)
    # ax1.set_xlim(10500, 3000)
    # ax1.set_ylim(5.50, -0.5)
    # ax1.tick_params(labelsize=18)
    # ax1.set_xlabel("$T_\mathrm{eff}$ [K] (Apogee)", fontsize=18)
    # ax1.set_ylabel("$\log{g}$ [dex] (Apogee)", fontsize=18)

    # plt.show()
    ###############################################################################################################

    # #####################multi process#############################################################################
    # result = joblib.Parallel(n_jobs=1, backend="multiprocessing")(
    #     joblib.delayed(_one_task)(_i, _j, CCF_specs) for (_i, _j) in zip([spec_list[101], spec_list[317], spec_list[627]], [params[101], params[317], params[627]]))
    # dump(result, 'test_CCF.dump')
    # #####################multiprocess#############################################################################

    # #####################Show Results#############################################################################
    # file_name = 'CCF_test_result/test_CCF_result_3000.csv'
    # tip = pd.read_csv(file_name)
    #
    # fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
    #
    # cm1 = plt.cm.get_cmap('jet')
    # im1 = ax1[0].scatter(tip['teff_ap'], tip['T_CCF']-tip["teff_ap"], s=30, c=tip['CCFmax'], marker=".",
    #                   alpha=1,
    #                   cmap=cm1, vmin=0.1, vmax=1.0)
    # position1 = fig1.add_axes([0.92, 0.12, 0.02, 0.76])  # 位置[左,下,右,上]
    # c1 = plt.colorbar(im1, cax=position1, orientation='vertical', )  # 方向
    # ax1[0].grid(True)
    # # ax1.set_xlim(-1000, 0.5)
    # ax1[0].set_ylim(-2000, 2000)
    # ax1[0].tick_params(labelsize=12)
    # ax1[0].set_xlabel("True Teff", fontsize=12)
    # ax1[0].set_ylabel("CCF Teff - True Teff", fontsize=12)
    # c1.set_label('CCFmax')
    #
    # im2 = ax1[1].scatter(tip['rv_lm'], tip['rv_CCF']-tip["rv_lm"], s=30, c=tip['CCFmax'], marker=".",
    #                   alpha=1,
    #                   cmap=cm1, vmin=0.1, vmax=1.0)
    # ax1[1].grid(True)
    # # ax1[1].set_xlim(-30, 30)
    # ax1[1].set_ylim(-30, 30)
    # ax1[1].tick_params(labelsize=12)
    # ax1[1].set_xlabel("LAMOST rv", fontsize=12)
    # ax1[1].set_ylabel("CCF rv - LAMOST rv", fontsize=12)
    #
    # plt.show()
    #
    # print(tip['obsid'][(tip['T_CCF'] - tip["teff_ap"])>1500])
    # #####################Show Results#############################################################################


    # ####################plot several observation spectra for check#######################
    # for i in [953, 1041, 2108]:
    #     flux_norm_array, flux_norm_err_array, snr_array, obsid_array, true_para_list = read_test_spec(spec_list[i], params[i])
    #
    #     plt.plot(wave, flux_norm_array[0], '-')
    #     plt.xlabel('wavelength')
    #     plt.ylabel('norm flux')
    #     plt.legend(('43410212', '405614078', '362707204'))
    # plt.show()
    # ####################plot several observation spectra for check#######################



    # ####################test positive_interp_then_norm###################################
    # flux_norm, flux_norm_err = positive_interp_then_norm(spec_list[317], wave)
    # flux_norm[flux_norm < 0] = 1
    # flux_norm[flux_norm > 3] = 1
    # plt.plot(wave, flux_norm, 'r-')
    # plt.plot(wave, flux_norm_err, 'g-')
    # plt.show()
    # ####################test positive_interp_then_norm###################################
