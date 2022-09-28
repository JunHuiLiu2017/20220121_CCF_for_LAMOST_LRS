import joblib
import numpy as np
from joblib import dump, load
from CCF_test_no_label import do_mock_spectra_CCF_flux_list


def _one_task_mock_binary(flux_norm_array, true_para, ccf_specs):
    spec_num = len(flux_norm_array[0])
    para_CCF = do_mock_spectra_CCF_flux_list(ccf_specs, flux_norm_array, [[0]]*spec_num, [[0]]*spec_num)
    para_CCF = para_CCF.tolist()
    para = [para_CCF[0], list(true_para[0])]
    print(para)
    return para


if __name__ == '__main__':
    model_dir = '/home/liujunhui/20220105model_single_binary/LamostBinary/'
    # load sp

    sp = joblib.load(model_dir + 'sp2022_05_29_13_09_44_Step_64013_1.dmp')

    wave = np.arange(3950, 5750, 1)
    CCF_wave = np.arange(3800, 6000, 1)

    CCF_specs = joblib.load(
        '/Users/liujunhui/PycharmProjects/20220121_CCF_for_LAMOST_LRS/2022_03_16_14_08_52_imitated_CCF_wl_3800_6000_1000_.dump')

    mock_binary_data = load('/Users/liujunhui/PycharmProjects/mock_spectra/20220629_main2_mock_binary_orbits_set_criteria/20220708_mock_binary_spectra_for_criteria.dump')
    mock_binary_params = mock_binary_data['params']
    mock_binary_spec_list = mock_binary_data['mock_flux_binary']

    #####################multi process#############################################################################
    result = joblib.Parallel(n_jobs=10, backend="multiprocessing")(
        joblib.delayed(_one_task_mock_binary)([_i], [_j], CCF_specs, sp) for (_i, _j) in zip(mock_binary_spec_list[:], mock_binary_params[:]))
    # dump(result, 'CCF_mock_binary/20220708_CCF_result_mock_binary_spectra_for_criteria.dump')
    #####################multi process#############################################################################