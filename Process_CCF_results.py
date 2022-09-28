import joblib
import pandas as pd
import os
from CCF_test_no_label import getFileName2


def process_CCF_result(file_path):
    data_list = joblib.load(file_path)
    return data_list


def screen_file(files_path, extension):                         # 函数功能为：筛选出文件夹下所有后缀名为.txt的文件
    path = files_path        	# 文件夹地址
    aim_list = []										# 创建一个空列表用于存放文件夹下所有后缀为.txt的文件名称
    file_list = os.listdir(path)                   	 	# 获取path文件夹下的所有文件，并生成列表
    for i in file_list:
        file_ext = os.path.splitext(i)              	# 分离文件前后缀，front为前缀名，ext为后缀名
        front, ext = file_ext							# 将前后缀分别赋予front和ext
        if ext == extension:                          		# 判断如果后缀名为.txt则将该文件名添加到txt_list的列表当中去
            aim_list.append(files_path+i)
    return aim_list


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


    # # Write the record of test CCF of mock binary to csv
    # # For the mock spectral sample with only two spectra for one object.!!!!!!!!!!!(20220425_test_CCF.dump)
    # test_result = joblib.load('CCF_mock_binary/20220629_CCF_correct_rv_elumock_binary_3000_logg3_teff_4k_7k_2_epoches_for_criteria.dump')
    # # for i in test_result:
    # #     print(i[1])
    # output_file = 'CCF_mock_binary/20220629_CCF_correct_rv_elumock_binary_3000_logg3_teff_4k_7k_2_epoches_for_criteria.csv'
    # with open(output_file, 'w+') as _file:
    #     _file.write('obsid(0),T_CCF,logg_CCF,M/H_CCF,alpha/M_CCF,rv_CCF,snr_CCF(0),CCFmax,' # 8
    #                 'teff1,teff2,logg1,logg2,mh,alpha_m,'  #6
    #                 'R1,R2,snr,period,just_num,mact1,mact2,logage,q,' #9
    #                 'gamma,q_dyn,rv1_1,rv2_1,rv1_2,rv2_2,phase1,phase2\n') #7
    #     for _i in test_result:
    #         single_record = '{:.0f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{:.3f},'\
    #                         '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},' \
    #                         '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},'\
    #                         '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
    #             _i[0][0], _i[0][1], _i[0][2], _i[0][3], _i[0][4], _i[0][5], _i[0][6], _i[0][7], \
    #             _i[1][0], _i[1][1], _i[1][2], _i[1][3], _i[1][4], _i[1][5],
    #              _i[1][6], _i[1][7], _i[1][8], 10**_i[1][9], _i[1][10], _i[1][11],_i[1][12], _i[1][13], _i[1][14],
    #             _i[1][15], _i[1][16], _i[1][17][0], _i[1][18][0], _i[1][17][1], _i[1][18][1], _i[1][19][0], _i[1][19][1])
    #         _file.write(single_record)



    # # # For the multi observed spectra from LAMOST  2022.6.22 for PhD Zhangzhixiang
    # # # For the multi observed spectra from LAMOST  !!!!!!!!!!!
    # # # In the step, I only get the first CCF result, so the _i[0][0].
    # test_result = joblib.load('./RV.dump')
    # # for i in test_result:
    # #     print(i[1])
    # output_file = './RV.csv'
    # with open(output_file, 'w+') as _file:
    #     _file.write('combined_obsid,T_CCF,logg_CCF,M/H_CCF,alpha/M_CCF,rv_CCF,snr_CCF,CCFmax\n') #7
    #     for _i in test_result:
    #         single_record = '{:.0f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{:.3f}\n'.format(
    #             _i[0][0], _i[0][1], _i[0][2], _i[0][3], _i[0][4], _i[0][5], _i[0][6], _i[0][7])
    #         _file.write(single_record)


    # Write the record of test CCF of mock binary to csv
    # For the mock spectral sample to get criteria. 2022.06.30
    # test_result = joblib.load('CCF_mock_binary/20220711_CCF_result_mock_binary_spectra_for_criteria.dump')
    # # for i in test_result:
    # #     print(i[1])
    # output_file = 'CCF_mock_binary/20220711_CCF_result_mock_binary_spectra_for_criteria.csv'
    # with open(output_file, 'w+') as _file:
    #     _file.write('obsid(0),T_CCF,logg_CCF,M/H_CCF,alpha/M_CCF,rv_CCF,snr_CCF(0),CCFmax,' # 8
    #                 'teff1,teff2,logg1,logg2,mh,alpha_m,'  #6
    #                 'R1,R2,snr,period,just_num,mact1,mact2,logage,q,' #9
    #                 'gamma,rv1_1,rv2_1\n') #7
    #     for _i in test_result:
    #         single_record = '{:.0f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{:.3f},'\
    #                         '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},' \
    #                         '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},'\
    #                         '{:.3f},{:.3f},{:.3f}\n'.format(
    #             _i[0][0], _i[0][1], _i[0][2], _i[0][3], _i[0][4], _i[0][5], _i[0][6], _i[0][7], \
    #             _i[1][0], _i[1][1], _i[1][2], _i[1][3], _i[1][4], _i[1][5],
    #              _i[1][6], _i[1][7], _i[1][8], 10**_i[1][9], _i[1][10], _i[1][11],_i[1][12], _i[1][13], _i[1][14],
    #             _i[1][15], _i[1][16], _i[1][17])
    #         _file.write(single_record)

    #################################################    After 2022.07.22    #####################################################



    # # # In the step, I only get the first CCF result, so the _i[0][0][0].
    # ## this section is only for one dump file.
    # test_result = joblib.load('./CCF_observed_binary/20220721_new_CCF_result_test_sample_observed_spectra_4997.dump')
    # # for i in test_result:
    # #     print(i[1])
    # output_file = './CCF_observed_binary/20220721_new_CCF_result_test_sample_observed_spectra_4997.csv'
    # with open(output_file, 'w+') as _file:
    #     _file.write('combined_obsid,T_CCF,logg_CCF,M/H_CCF,alpha/M_CCF,rv_CCF,snr_CCF,CCFmax,T_sp,logg_sp,M/H_sp,'
    #                 'alpha/M_sp\n') #7
    #     for _i in test_result:
    #         single_record = '{:.0f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{:.3f},' \
    #                         '{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
    #             _i[0][0][0], _i[0][0][1], _i[0][0][2], _i[0][0][3], _i[0][0][4], _i[0][0][5], _i[0][0][6], _i[0][0][7],
    #             _i[1][0][1], _i[1][0][2], _i[1][0][3], _i[1][0][4])
    #         _file.write(single_record)


#####################################################   2022.09.25 (for Total CCF result)  ##########################################
    # # In the step, I only get the first CCF result, so the _i[0][0][0].
    results_path = '/Users/liujunhui/PycharmProjects/20220121_CCF_for_LAMOST_LRS/CCF_observed_binary_TOTAL/'
    results_dump_paths = screen_file(results_path, '.dump')
    print(results_dump_paths)

    output_file = '/Users/liujunhui/PycharmProjects/20220121_CCF_for_LAMOST_LRS/CCF_observed_binary_TOTAL' \
                  '/CCF_result_sum.csv'
    with open(output_file, 'w+') as _file:
        _file.write('combined_obsid,T_CCF,logg_CCF,M/H_CCF,alpha/M_CCF,rv_CCF,snr_CCF,CCFmax,T_sp,logg_sp,M/H_sp,'
                    'alpha/M_sp\n')
        for one_path in results_dump_paths:
            test_result = joblib.load(one_path)
            # print(test_result)
            for _i in test_result:
                single_record = '{:.0f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.\
                    format(_i[0][0][0], _i[0][0][1], _i[0][0][2], _i[0][0][3], _i[0][0][4], _i[0][0][5], _i[0][0][6], \
                        _i[0][0][7],_i[1][0][1], _i[1][0][2], _i[1][0][3], _i[1][0][4])
                _file.write(single_record)

    whole_data = pd.read_csv('./20220915_LAMOST_DR9_snrg30_4289736_G.csv')
    CCF_result = pd.read_csv(output_file)
    df2 = pd.merge(whole_data, CCF_result, on=['combined_obsid'])
    df2.sort_values(by='combined_obsid')
    df2.to_csv('/Users/liujunhui/PycharmProjects/20220121_CCF_for_LAMOST_LRS/CCF_observed_binary_TOTAL'
               '/CCF_result_sum_all.csv', index=False)





