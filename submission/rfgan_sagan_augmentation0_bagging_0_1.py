import matplotlib.pyplot as plt
import os
import pandas as pd

root1 = 'experiments/rfgan'
exp1 = 'rfgan_lossgan1_bagging0_augmentation0_5heads_usemaskD_sagan'
exp2 = 'rfgan_lossgan1_bagging1_augmentation0_5heads_usemaskD_sagan'

df1 = pd.read_csv(os.path.join(root1, exp1, exp1 + '.csv'))
df2 = pd.read_csv(os.path.join(root1, exp2, exp2 + '.csv'))

n = min(df1.shape[0], df2.shape[0])
print(n)
df1 = df1.iloc[0:n]
df2 = df2.iloc[0:n]

print(df1['fid_score'].min())
print(df2['fid_score'].min())

epochs = [x * 4 for x in range(n)]

colors = ['green', 'red', 'orange', 'blue']

fig, axs = plt.subplots(1, 4, sharex=False, sharey=False, figsize=(16, 5))

axs[0].plot(epochs, df1['dx_max_min'], color=colors[0], label='RFGan + no bagging')
axs[0].plot(epochs, df2['dx_max_min'], color=colors[1], label='RFGan + bagging')
axs[0].set_xlabel('Epoch', fontsize='xx-large')
axs[0].set_ylabel(r'$W_{F_d(x)}$', fontsize='xx-large')

axs[1].plot(epochs, df1['dgz_max_min'], color=colors[0])
axs[1].plot(epochs, df2['dgz_max_min'], color=colors[1])
axs[1].set_xlabel('Epoch', fontsize='xx-large')
axs[1].set_ylabel(r'$W_{F_d(G(z))}$', fontsize='xx-large')

n1 = 5
axs[2].plot(epochs[n1:], df1['fid_score'][n1:], color=colors[0])
axs[2].plot(epochs[n1:], df2['fid_score'][n1:], color=colors[1])
axs[2].set_xlabel('Epoch', fontsize='xx-large')
axs[2].set_ylabel('FID', fontsize='xx-large')

axs[3].plot(epochs, df1['lossd_mean'], color=colors[0])
axs[3].plot(epochs, df2['lossd_mean'], color=colors[1])
axs[3].set_xlabel('Epoch', fontsize='xx-large')
axs[3].set_ylabel(r'$L_{F_d}$', fontsize='xx-large')


handles, labels = axs[0].get_legend_handles_labels()
axs[3].legend(handles=handles, labels=labels, loc='lower center',
              bbox_to_anchor=(-1.5, -0.45), fancybox=False, shadow=False,
              ncol=4, fontsize='xx-large')

plt.subplots_adjust(top=.9, bottom=0.3, right=0.99, left=0.05,
                    hspace=0, wspace=0.32)
plt.savefig('submission/rfgan_sagan_augmentation0_bagging_0_1.pdf')
