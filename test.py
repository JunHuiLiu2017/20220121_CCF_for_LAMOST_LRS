from joblib import load
import matplotlib.pyplot as plt
import numpy as np

A = load('2.dump')

fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))

# Apogee data
cm2 = plt.cm.get_cmap('Blues')
im2 = ax1.scatter(A['p_test'][:, 0], A["p_CCF"][:, 0], s=30, c=A['p_test'][:, 5], marker="o", alpha=1, cmap=cm2, vmin=30, vmax=200)
position2=fig1.add_axes([0.92, 0.12, 0.02, 0.76])#位置[左,下,右,上]
c2=plt.colorbar(im2,cax=position2,orientation='vertical')#方向
#c2.set_label("[Fe/H] [dex]", fontsize = 18)
ax1.grid(True)
ax1.set_xlim(3000, 10500)
ax1.set_ylim(3000, 10500)
ax1.tick_params(labelsize=18)
ax1.set_xlabel("True Teff", fontsize = 18)
ax1.set_ylabel("CCF Teff", fontsize = 18)

#fig.tight_layout()
plt.show()

print(A['rv_CCF'])

plt.plot(A['p_test'][:, 4], A['rv_CCF'][:, 0], '.')
plt.plot(np.arange(-400,400), np.arange(-400,400), '-')
plt.xlabel("LAMOST RV", fontsize = 18)
plt.ylabel("CCF RV", fontsize = 18)
plt.show()