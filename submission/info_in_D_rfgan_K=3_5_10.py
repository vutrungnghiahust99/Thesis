import matplotlib.pyplot as plt
import os
import pandas as pd

root1 = 'experiments/rfgan'
exp1 = 'rfgan_lossgan1_bagging1_augmentation0_3heads_usemaskD'
exp2 = 'rfgan_lossgan1_bagging1_augmentation0_5heads_usemaskD'
exp3 = 'rfgan_lossgan1_bagging1_augmentation0_10heads_usemaskD'


df1 = pd.read_csv(os.path.join(root1, exp1, exp1 + '.csv'))
df2 = pd.read_csv(os.path.join(root1, exp2, exp2 + '.csv'))
df3 = pd.read_csv(os.path.join(root1, exp3, exp3 + '.csv'))

n = min(df1.shape[0], df2.shape[0], df3.shape[0])

print(n)
df1 = df1.iloc[0:n]
df2 = df2.iloc[0:n]
df3 = df3.iloc[0:n]

epochs = [x * 4 for x in range(n)]

colors = ['green', 'red', 'blue']

fig, axs = plt.subplots(1, 5, sharex=False, sharey=False, figsize=(20, 5))

axs[0].plot(epochs, df1['dx_max_min'], color=colors[0], label='K=3')
axs[0].plot(epochs, df2['dx_max_min'], color=colors[1], label='K=5')
axs[0].plot(epochs, df3['dx_max_min'], color=colors[2], label='K=10')
axs[0].set_xlabel('Epoch', fontsize='xx-large')
axs[0].set_ylabel(r'$W_{F_D(x)}$', fontsize='x-large')

axs[1].plot(epochs, df1['dgz_max_min'], color=colors[0])
axs[1].plot(epochs, df2['dgz_max_min'], color=colors[1])
axs[1].plot(epochs, df3['dgz_max_min'], color=colors[2])
axs[1].set_xlabel('Epoch', fontsize='xx-large')
axs[1].set_ylabel(r'$W_{F_D(G(z))}$', fontsize='x-large')


axs[2].plot(epochs, df1['dgz_mean'], color=colors[0])
axs[2].plot(epochs, df2['dgz_mean'], color=colors[1])
axs[2].plot(epochs, df3['dgz_mean'], color=colors[2])
axs[2].set_xlabel('Epoch', fontsize='xx-large')
axs[2].set_ylabel(r'$F_D(G(z))$', fontsize='x-large')

axs[3].plot(epochs, df1['dx_mean'], color=colors[0])
axs[3].plot(epochs, df2['dx_mean'], color=colors[1])
axs[3].plot(epochs, df3['dx_mean'], color=colors[2])
axs[3].set_xlabel('Epoch', fontsize='xx-large')
axs[3].set_ylabel(r'$F_D(x)$', fontsize='x-large')

n1 = 20
axs[4].plot(epochs[n1:], df1['fid_score'][n1:], color=colors[0])
axs[4].plot(epochs[n1:], df2['fid_score'][n1:], color=colors[1])
axs[4].plot(epochs[n1:], df3['fid_score'][n1:], color=colors[2])
axs[4].set_xlabel('Epoch', fontsize='xx-large')
axs[4].set_ylabel('FID', fontsize='x-large')

handles, labels = axs[0].get_legend_handles_labels()
axs[3].legend(handles=handles, labels=labels, loc='lower center',
              bbox_to_anchor=(-0.81, -0.45), fancybox=False, shadow=False,
              ncol=4, fontsize='xx-large')

plt.subplots_adjust(top=.9, bottom=0.3, right=0.99, left=0.05,
                    hspace=0, wspace=0.32)
plt.savefig('submission/info_in_D_rfgan_K=3_5_10.pdf')
