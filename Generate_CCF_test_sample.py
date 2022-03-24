from joblib import load, dump
import random


# generate CCF test sample.
full_sample = load('/Users/liujunhui/Downloads/2022_01_03_13_54_24_74768Apdr16_para_dump_ap_lm.dpl')

random_index = random.sample(range(0, 74768), 3000)

select_specs = []
select_params = []
for _i in random_index:
    select_specs.append(full_sample['spec_list'][_i])
    select_params.append(list(full_sample['params'][_i]))

CCF_test_sample={'params':select_params,'spec_list':select_specs}
dump(CCF_test_sample, 'small_sample_3000.dump')


