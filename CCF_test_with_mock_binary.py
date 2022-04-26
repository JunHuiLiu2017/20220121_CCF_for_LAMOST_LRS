import joblib
import numpy as np
from joblib import dump, load
from CCF_test_no_label import do_CCF_flux_list


def _one_task_mock_binary(flux_norm_array, true_para, ccf_specs):
    spec_num = len(flux_norm_array[0])
    para_CCF = do_CCF_flux_list(ccf_specs, flux_norm_array, snr_array=[[0]]*spec_num, obsid_array=[[0]]*spec_num)
    para_CCF = para_CCF.tolist()
    para = [para_CCF[0], list(true_para[0])]
    print(para)
    return para


if __name__ == '__main__':
    wave = np.arange(3950, 5750, 1)
    CCF_wave = np.arange(3800, 6000, 1)

    CCF_specs = joblib.load(
        '/Users/liujunhui/PycharmProjects/20220121_CCF_for_LAMOST_LRS/2022_03_16_14_08_52_imitated_CCF_wl_3800_6000_1000_.dump')

    mock_binary_data = load('/Users/liujunhui/PycharmProjects/mock_spectra/mock_binary_data_logg_3_Teff_4000_7000_2_epoches.dump')
    mock_binary_params = mock_binary_data['params']
    mock_binary_spec_list = mock_binary_data['mock_flux_binary']

    #####################multi process#############################################################################
    result = joblib.Parallel(n_jobs=1, backend="multiprocessing")(
        joblib.delayed(_one_task_mock_binary)([_i], [_j], CCF_specs) for (_i, _j) in zip(mock_binary_spec_list[:10], mock_binary_params[:10]))
    dump(result, './CCF_mock_binary/20220425_test_CCF.dump')
    #####################multi process#############################################################################
