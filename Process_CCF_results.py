import joblib
import pandas as pd

from CCF_test_no_label import getFileName2


def process_CCF_result(file_path):
    data_list = joblib.load(file_path)
    return data_list


if __name__ == '__main__':

    # Write the record of test CCF to csv
    # test_result = joblib.load('CCF_test_result/test_CCF_3000.dump')
    # with open('CCF_test_result/test_CCF_result_3000.csv', 'w+') as _file:
    #     _file.write('obsid,T_CCF,logg_CCF,M/H_CCF,alpha/M_CCF,rv_CCF,snr_CCF,CCFmax,teff_ap,logg_ap,feh_ap,a_M_ap,Vsini_ap,'
    #                 'Vmicro_ap,teff_lm,logg_lm,feh_lm,rv_lm,fps\n')
    #     for _i in test_result:
    #         single_record = '{:.0f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},' \
    #                         '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{}\n'.format(_i[0][0], _i[0][1], _i[0][2], _i[0][3],
    #                                                                          _i[0][4], _i[0][5], _i[0][6], _i[0][7],_i[1][0],
    #                                                                          _i[1][1], _i[1][2], _i[1][3], _i[1][4],
    #                                                                          _i[1][5], _i[1][6], _i[1][7], _i[1][8],
    #                                                                          _i[1][9], _i[1][10])
    #         _file.write(single_record)

    # Write the record of test CCF of mock binary to csv
    # For the mock spectral sample with only one spectra for one object.!!!!!!!!!!!
    # test_result = joblib.load('CCF_mock_binary/test_mock_binary_result_logg_3_Teff_4000_7000.dump')
    # output_file = 'CCF_mock_binary/test_mock_binary_result_logg_3_Teff_4000_7000.csv'
    # with open(output_file, 'w+') as _file:
    #     _file.write('obsid(0),T_CCF,logg_CCF,M/H_CCF,alpha/M_CCF,rv_CCF,snr_CCF(0),CCFmax,teff1,teff2,logg1,logg2,mh,'
    #                 'alpha_m,R1,R2,rv1,rv2,snr,just_num,mact1,mact2,logage,q\n')
    #     for _i in test_result:
    #         single_record = '{:.0f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},' \
    #                         '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
    #             _i[0][0], _i[0][1], _i[0][2], _i[0][3], _i[0][4], _i[0][5], _i[0][6], _i[0][7], _i[1][0], _i[1][1],
    #             _i[1][2], _i[1][3], _i[1][4], _i[1][5], _i[1][6], _i[1][7], _i[1][8], _i[1][9], _i[1][10], _i[1][11],
    #             _i[1][12], _i[1][13], _i[1][14], _i[1][15])
    #         _file.write(single_record)

    # Write the record of test CCF of mock binary to csv
    # For the mock spectral sample with only two spectra for one object.!!!!!!!!!!!(20220425_test_CCF.dump)
    test_result = joblib.load('CCF_mock_binary/20220425_test_CCF.dump')
    output_file = 'CCF_mock_binary/20220425_test_CCF.csv'
    with open(output_file, 'w+') as _file:
        _file.write('obsid(0),T_CCF,logg_CCF,M/H_CCF,alpha/M_CCF,rv_CCF,snr_CCF(0),CCFmax,' # 8
                    'teff1,teff2,logg1,logg2,mh,alpha_m,'  #6
                    'R1,R2,snr,period,just_num,mact1,mact2,logage,q,' #9
                    'gamma,rv1_1,rv2_1,rv1_2,rv2_2,phase1,phase2\n') #7
        for _i in test_result:
            single_record = '{:.0f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{:.3f},'\
                            '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},' \
                            '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},'\
                            '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
                _i[0][0], _i[0][1], _i[0][2], _i[0][3], _i[0][4], _i[0][5], _i[0][6], _i[0][7], \
                _i[1][0], _i[1][1], _i[1][2], _i[1][3], _i[1][4], _i[1][5],
                 _i[1][6], _i[1][7], _i[1][8], 10**_i[1][9], _i[1][10], _i[1][11],_i[1][12], _i[1][13], _i[1][14],
                _i[1][15], _i[1][16][0], _i[1][17][0], _i[1][16][1], _i[1][17][1], _i[1][18][0], _i[1][18][1])
            _file.write(single_record)