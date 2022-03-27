import numpy as np
from joblib import load, dump
import random


# generate CCF test sample.
full_sample = load('/Users/liujunhui/Downloads/2022_01_03_13_54_24_74768Apdr16_para_dump_ap_lm.dpl')
Filter1 = load("/Users/liujunhui/Downloads/2021_12_26_00_03_21filter1_lessthan4.dmp")
print(Filter1)
Filter2 = load("/Users/liujunhui/Downloads/2021_12_27_22_25_35filter2_lessthan4.dmp")
print(len(Filter2[Filter2 == True]))
Filter3 = load("/Users/liujunhui/Downloads/2021_12_28_10_39_17filter3_lessthan4.dmp")
print(len(Filter3[Filter3 == True]))

random_index = random.sample(range(0, 62577), 3000)

select_specs = []
select_params = []
for _i in random_index:
    select_specs.append(np.array(full_sample['spec_list'])[Filter1][Filter2][Filter3][_i])
    select_params.append(list(np.array(full_sample['params'])[Filter1][Filter2][Filter3][_i]))

CCF_test_sample={'params':select_params,'spec_list':select_specs}
dump(CCF_test_sample, 'small_sample_3000.dump')
