import joblib
from CCF_test_no_label import getFileName2


def process_CCF_result(file_path):
    data_list = joblib.load(file_path)
    return data_list


if __name__ == '__main__':
    # work_path = './CCF_results/'
    # all_file, all_file_path = getFileName2(work_path, '.dump')
    #
    # with open(work_path + 'CCF_result.csv', 'w+') as _file:
    #     _file.write('obsid,T_CCF,logg_CCF,M/H_CCF,alpha/M_CCF,rv_CCF,snr_CCF\n')
    #     for _i in all_file_path:
    #         data = process_CCF_result(_i)
    #         for _j in data:
    #             for _k in _j:
    #                 if len(_k) == 7:
    #                     single_record = '{:.0f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.1f}\n'.format(_k[0], _k[1], _k[2],
    #                                                                                               _k[3], _k[4], _k[5],
    #                                                                                               _k[6])
    #                     _file.write(single_record)
    #                 elif len(_k) != 7:
    #                     print(_j)


    #Write the record to csv
    test_result = joblib.load('CCF_test_result/test_CCF_2000.dump')
    with open('CCF_test_result/test_CCF_result_2000.csv', 'w+') as _file:
        _file.write('obsid,T_CCF,logg_CCF,M/H_CCF,alpha/M_CCF,rv_CCF,snr_CCF,CCFmax,teff_ap,logg_ap,feh_ap,a_M_ap,Vsini_ap,'
                    'Vmicro_ap,teff_lm,logg_lm,feh_lm,rv_lm,fps\n')
        for _i in test_result:
            single_record = '{:.0f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},' \
                            '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{}\n'.format(_i[0][0], _i[0][1], _i[0][2], _i[0][3],
                                                                             _i[0][4], _i[0][5], _i[0][6], _i[0][7],_i[1][0],
                                                                             _i[1][1], _i[1][2], _i[1][3], _i[1][4],
                                                                             _i[1][5], _i[1][6], _i[1][7], _i[1][8],
                                                                             _i[1][9], _i[1][10])
            _file.write(single_record)