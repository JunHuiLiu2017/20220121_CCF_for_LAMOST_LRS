import joblib
import pandas as pd
from slam.diagnostic import compare_labels
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib
import random
import laspec
import numpy as np
from laspec.normalization import normalize_spectrum_general
from laspec.ccf import wxcorr_rvgrid
import time
from joblib import dump, load
from laspec import mrs


def test_lamost_specs(fits_list, test_size, wave=np.arange(3950, 5750, 1)):
    for j in range(test_size):
        test_spec = fits_list[j].interp_then_norm(wave)[0]
        plt.plot(wave, test_spec, 'r-')

def test_lamost_specs2(flux_list, test_size, wave=np.arange(3950, 5750, 1)):
    for j in range(test_size):
        test_spec = flux_list[j]
        plt.plot(wave, test_spec, 'b-')
    plt.show()

def do_CCF_test(CCF_specs, fits_list, fits_params, test_size, wave=np.arange(3950, 5750, 1)):
    p_test = np.zeros((test_size, 6), dtype=float)
    p_CCF = np.zeros((test_size, 4), dtype=float)
    rv_CCF = np.zeros((test_size, 1), dtype=float)
    for j in range(test_size):
        if j % 1 == 0:
            print(j, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
        test_spec = fits_list[j].interp_then_norm(wave, rv=0)[0]
        snr = fits_list[j].snr
        p_test[j] = [fits_params['teff_ap'][j], fits_params['logg_ap'][j], fits_params['feh_ap'][j],
                     fits_params['a_M_ap'][j], fits_params["rv_lm"][j], snr]
        maxvalues = []
        rvs = []
        for i in CCF_specs['flux_norm_regli_CCF']:
            rvgrid, ccf = wxcorr_rvgrid(wave, test_spec, CCF_wave, i, rv_grid=np.linspace(-500, 500, 100))

            maxvalues.append(max(ccf))
            rvs.append(rvgrid[np.argmax(ccf)])
            # plt.plot(CCF_wave, i, 'r-')
            # plt.plot(wave, test_spec, 'b-')

        p_CCF[j] = CCF_specs['p_regli_CCF'][np.argmax(maxvalues)]
        rv_CCF[j] = rvs[np.argmax(maxvalues)]
        # plt.plot(CCF_wave, CCF_specs['flux_norm_regli_CCF'][np.argmax(maxvalues)], 'r-')
        # plt.plot(wave, test_spec, 'b-')
        # plt.show()

    return {'p_test': p_test,
            'p_CCF': p_CCF,
            'rv_CCF': rv_CCF}



wave = np.arange(3950, 5750, 1)
CCF_wave = np.arange(3800, 6000, 1)

CCF_specs = joblib.load(
    '/Users/liujunhui/PycharmProjects/20220121_CCF_for_LAMOST_LRS/2022_01_23_23_47_23_imitated_CCF_wl_3800_6000_300_.dump')

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'  # 将刻度显示成朝里
matplotlib.rcParams['ytick.right'] = 'True'
matplotlib.rcParams['xtick.top'] = 'True'  # 设置x轴上方横线显示刻度
matplotlib.rcParams['grid.linestyle'] = '--'  # 设置标度线的风格

###############################################################################################################
# show the distribution of CCF spectra.
fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
cm2 = plt.cm.get_cmap('jet')
im2 = ax1.scatter(CCF_specs['p_regli_CCF'][:, 0], CCF_specs['p_regli_CCF'][:, 1], s=30,
                  c=CCF_specs['p_regli_CCF'][:, 2], marker="o", edgecolor='k', alpha=.8, cmap=cm2, vmin=-2, vmax=0.8)
position2 = fig1.add_axes([0.92, 0.12, 0.02, 0.76])  # 位置[左,下,右,上]
c2 = plt.colorbar(im2, cax=position2, orientation='vertical')  # 方向
# c2.set_label("[Fe/H] [dex]", fontsize = 18)
ax1.grid(True)
ax1.set_xlim(10500, 3000)
ax1.set_ylim(5.50, -0.5)
ax1.tick_params(labelsize=18)
ax1.set_xlabel("$T_\mathrm{eff}$ [K] (Apogee)", fontsize=18)
ax1.set_ylabel("$\log{g}$ [dex] (Apogee)", fontsize=18)

#plt.show()
###############################################################################################################


test_data = load('/Users/liujunhui/PycharmProjects/20220121_CCF_for_LAMOST_LRS/small_sample_1000.dump')
params = test_data['params'][:20]
spec_list = test_data['spec_list'][:20]

results = do_CCF_test(CCF_specs=CCF_specs, fits_list=spec_list, fits_params=params, test_size=len(params))
dump(results, '2.dump')

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
cm2 = plt.cm.get_cmap('Blues')
im2 = ax1.scatter(results['p_test'][:, 0], results["p_CCF"][:, 0], s=30, c=results['p_test'][:, 5], marker="o", alpha=1,
                  cmap=cm2, vmin=30, vmax=200)
position2 = fig1.add_axes([0.92, 0.12, 0.02, 0.76])  # 位置[左,下,右,上]
c2 = plt.colorbar(im2, cax=position2, orientation='vertical')  # 方向
# c2.set_label("[Fe/H] [dex]", fontsize = 18)
ax1.grid(True)
ax1.set_xlim(3000, 10500)
ax1.set_ylim(3000, 10500)
ax1.tick_params(labelsize=18)
ax1.set_xlabel("True Teff", fontsize=18)
ax1.set_ylabel("CCF Teff", fontsize=18)
plt.show()
