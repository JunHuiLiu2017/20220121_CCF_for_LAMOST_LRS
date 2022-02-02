import joblib
import numpy as np
from laspec.ccf import wxcorr_rvgrid
import time

from laspec import mrs


def do_CCF(ccf_specs, work_path, fits_path, test_size=1, wave=np.arange(3950, 5750, 1)):
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

    para_CCF = np.zeros((test_size, 7), dtype=float)

    for j in range(test_size):
        if j % 1 == 0:
            print(j, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
        fits  = mrs.MrsSpec.from_lrs(work_path + fits_path)
        test_spec = fits.interp_then_norm(wave, rv=0)[0]
        snr = fits.snr
        obsid = fits.obsid
        maxvalues = []
        rvs = []
        for i in ccf_specs['flux_norm_regli_CCF']:
            rvgrid, ccf = wxcorr_rvgrid(wave, test_spec, CCF_wave, i, rv_grid=np.linspace(-500, 500, 100))
            maxvalues.append(max(ccf))
            rvs.append(rvgrid[np.argmax(ccf)])

        para_CCF[j][0] = obsid
        para_CCF[j][1:5] = CCF_specs['p_regli_CCF'][np.argmax(maxvalues)]
        para_CCF[j][5] = snr
        para_CCF[j][6] = rvs[np.argmax(maxvalues)]
        # plt.plot(CCF_wave, CCF_specs['flux_norm_regli_CCF'][np.argmax(maxvalues)], 'r-')
        # plt.plot(wave, test_spec, 'b-')
        # plt.show()

    return {'para_CCF': para_CCF}

if __name__ == '__main__':
    wave = np.arange(3950, 5750, 1)
    CCF_wave = np.arange(3800, 6000, 1)

    CCF_specs = joblib.load(
        '/Users/liujunhui/PycharmProjects/20220121_CCF_for_LAMOST_LRS/2022_01_23_23_47_23_imitated_CCF_wl_3800_6000_300_.dump')

    work_path = '/Users/liujunhui/Desktop/2021workMac/202111012totallynew/test_data_group/Test_data_files/'
    fits_path = 'spec-55859-F5902_sp03-232.fits.gz'
    start = time.time
    results = do_CCF(ccf_specs=CCF_specs, work_path=work_path, fits_path=fits_path, test_size=1, wave=np.arange(3950, 5750, 1))
    print(results['para_CCF'][0])