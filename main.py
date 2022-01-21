import joblib
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

wave = np.arange(3950, 5750, 1)

# Creat model spectra for CCF

# whole_sample = joblib.load('C:/Users/hp/Desktop/20211109laspec_tutorial/2021_12_30_14_30_54_imitated_26500_regli.dump')
# randomnum = np.random.randint(0,high=26500,size=300,dtype='int')
# spec_num = len(randomnum)
# flux_regli = np.zeros((spec_num, len(wave)), dtype = float)
# flux_norm_regli = np.zeros((spec_num, len(wave)), dtype = float)
# p_regli = np.zeros((spec_num, 4), dtype = float)
# snr_regli = np.zeros((spec_num, 1), dtype = float)
#
# for i in range(0, spec_num):
#     flux_regli[i] = whole_sample['flux_regli'][randomnum[i]]
#     flux_norm_regli[i] =  normalize_spectrum_general(wave, whole_sample['flux_regli'][randomnum[i]])[0]
#     p_regli[i] =  whole_sample['p_regli'][randomnum[i]]
#     snr_regli[i] =  whole_sample['snr_regli'][randomnum[i]]
#
# database={'wave':wave, 'flux_regli_CCF':flux_regli, 'flux_norm_regli_CCF':flux_norm_regli, 'p_regli_CCF':p_regli, 'snr_regli_CCF': snr_regli}
# timevalue = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
#
# dump(database,'./CCF_records/'+timevalue+'_imitated_CCF'+'_'+str(spec_num)+'_regli.dump')

CCF_specs = joblib.load('C:/Users/hp/Desktop/20211109laspec_tutorial/CCF/CCF_datas/2022_01_19_14_10_50_imitated_CCF_300_regli.dump')

matplotlib.rcParams['xtick.direction']='in'
matplotlib.rcParams['ytick.direction']='in'#将刻度显示成朝里
matplotlib.rcParams['ytick.right']='True'
matplotlib.rcParams['xtick.top']='True'#设置x轴上方横线显示刻度
matplotlib.rcParams['grid.linestyle']='--'#设置标度线的风格

fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 8))

# Apogee data
cm2 = plt.cm.get_cmap('jet')
im2 = ax1.scatter(CCF_specs['p_regli_CCF'][:, 0], CCF_specs['p_regli_CCF'][:, 1], s=30, c=CCF_specs['p_regli_CCF'][:, 2], marker="o", edgecolor ='k', alpha=.8, cmap=cm2, vmin=-2, vmax=0.8)
position2=fig1.add_axes([0.92, 0.12, 0.02, 0.76])#位置[左,下,右,上]
c2=plt.colorbar(im2,cax=position2,orientation='vertical')#方向
#c2.set_label("[Fe/H] [dex]", fontsize = 18)
ax1.grid(True)
ax1.set_xlim(10500, 3000)
ax1.set_ylim(5.50,-0.5)
ax1.tick_params(labelsize=18)
ax1.set_xlabel("$T_\mathrm{eff}$ [K] (Apogee)", fontsize = 18)
ax1.set_ylabel("$\log{g}$ [dex] (Apogee)", fontsize = 18)

#fig.tight_layout()
#plt.show()

##############################################################################################


def do_CCF(test_size, wave, ):
    p_test = np.zeros((test_size, 4), dtype = float)
    p_CCF = np.zeros((test_size, 4), dtype = float)
    rv_CCF = np.zeros((test_size, 1), dtype = float)
    for j in range(test_size):
        if j%500 == 0:
            print(j, time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())))
        test_spec = normalize_spectrum_general(wave, test_specs['flux_regli_test'][j])[0]
        p_test[j] = test_specs['p_test'][j] #  the True lable of test spactra
        maxvalues = []
        rvs = []
        for i in CCF_specs['flux_norm_regli_CCF']:
            rvgrid, ccf = wxcorr_rvgrid(wave, test_spec, wave, i, rv_grid=np.linspace(-105, 100, 20))
            maxvalues.append(max(ccf))
            rvs.append(rvgrid[np.argmax(ccf)])
        p_CCF[j] = CCF_specs['p_CCF'][np.argmax(maxvalues)]
        rv_CCF[j] = rvs[np.argmax(maxvalues)]



test_data = load('D:/2020_workspace/2021work/20211109laspec_tutorial/2021_12_24_22_49_08_74768Apdr16_para_dump_ap_lm.dpl')

params = test_data['params'][::100]
spec_list = test_data['spec_list'][::100]

print(params)
#print(spec_list)

