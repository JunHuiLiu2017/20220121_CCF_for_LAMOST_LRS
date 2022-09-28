import pandas as pd

whole_data = pd.read_csv('./20220915_LAMOST_DR9_snrg30_4289736_G.csv')
CCF_result = pd.read_csv('./CCF_result_sum.csv')

df2 = pd.merge(whole_data, CCF_result, on=['combined_obsid'])

df2.to_csv('./CCF_result_sum_all.csv', index=False)