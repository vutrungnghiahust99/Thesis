import matplotlib.pyplot as plt
import os
import pandas as pd

root1 = 'experiments/rfgan'
exp1 = 'rfgan_lossgan1_bagging0_augmentation0_10heads_usemaskD'
exp2 = 'rfgan_lossgan1_bagging1_augmentation0_10heads_usemaskD'
exp3 = 'experiments/baseline/baseline_mheads/baselinemheads_lossgan1_augmentation0/baselinemheads_lossgan1_augmentation0.csv'
exp4 = 'experiments/baseline/baseline_hgan/baselinehgan_losshgan_augmentation0/baselinehgan_losshgan_augmentation0.csv'


root2 = 'experiments/rfgan/augmentation'
exp5 = 'rfgan_lossgan1_bagging0_augmentation1_10heads_usemaskD'
exp6 = 'rfgan_lossgan1_bagging1_augmentation1_10heads_usemaskD'
exp7 = 'experiments/baseline/baseline_mheads/baselinemheads_lossgan1_augmentation1/baselinemheads_lossgan1_augmentation1.csv'
exp8 = 'experiments/baseline/baseline_hgan/baselinehgan_losshgan_augmentation1/baselinehgan_losshgan_augmentation1.csv'


df1 = pd.read_csv(os.path.join(root1, exp1, exp1 + '.csv'))
df2 = pd.read_csv(os.path.join(root1, exp2, exp2 + '.csv'))
df3 = pd.read_csv(exp3)
df4 = pd.read_csv(exp4)

df5 = pd.read_csv(os.path.join(root2, exp5, exp5 + '.csv'))
df6 = pd.read_csv(os.path.join(root2, exp6, exp6 + '.csv'))
df7 = pd.read_csv(exp7)
df8 = pd.read_csv(exp8)

n = min(
    df1.shape[0], df2.shape[0], df3.shape[0], df4.shape[0],
    df5.shape[0], df6.shape[0], df7.shape[0], df8.shape[0])
print(n)
df1 = df1.iloc[0:n]
df2 = df2.iloc[0:n]
df3 = df3.iloc[0:n]
df4 = df4.iloc[0:n]

df5 = df5.iloc[0:n]
df6 = df6.iloc[0:n]
df7 = df7.iloc[0:n]
df8 = df8.iloc[0:n]
epochs = [x * 4 for x in range(n)]

colors = ['green', 'red', 'orange', 'blue']

fig, axs = plt.subplots(1, 4, sharex=False, sharey=False, figsize=(16, 5))

axs[0].plot(epochs, df1['dx_max_min'], color=colors[0], label='RFGan + no bagging')
axs[0].plot(epochs, df2['dx_max_min'], color=colors[1], label='RFGan + bagging')
axs[0].plot(epochs, df3['dx_max_min'], color=colors[2], label='Multiple D')
axs[0].plot(epochs, df4['dx_max_min'], color=colors[3], label='HGan')
axs[0].set_xlabel('Epoch', fontsize='xx-large')
axs[0].set_ylabel(r'$W_{F_D(x)}$', fontsize='x-large')
axs[0].set_title('no augmentation')

axs[1].plot(epochs, df1['dgz_max_min'], color=colors[0])
axs[1].plot(epochs, df2['dgz_max_min'], color=colors[1])
axs[1].plot(epochs, df3['dgz_max_min'], color=colors[2])
axs[1].plot(epochs, df4['dgz_max_min'], color=colors[3])
axs[1].set_xlabel('Epoch', fontsize='xx-large')
axs[1].set_ylabel(r'$W_{F_D(G(z))}$', fontsize='x-large')
axs[1].set_title('no augmentation')


axs[2].plot(epochs, df5['dx_max_min'], color=colors[0])
axs[2].plot(epochs, df6['dx_max_min'], color=colors[1])
axs[2].plot(epochs, df7['dx_max_min'], color=colors[2])
axs[2].plot(epochs, df8['dx_max_min'], color=colors[3])
axs[2].set_xlabel('Epoch', fontsize='xx-large')
axs[2].set_ylabel(r'$W_{F_D(x)}$', fontsize='x-large')
axs[2].set_title("augmentation")

axs[3].plot(epochs, df5['dgz_max_min'], color=colors[0])
axs[3].plot(epochs, df6['dgz_max_min'], color=colors[1])
axs[3].plot(epochs, df7['dgz_max_min'], color=colors[2])
axs[3].plot(epochs, df8['dgz_max_min'], color=colors[3])
axs[3].set_xlabel('Epoch', fontsize='xx-large')
axs[3].set_ylabel(r'$W_{F_D(G(z))}$', fontsize='x-large')
axs[3].set_title("augmentation")

handles, labels = axs[0].get_legend_handles_labels()
axs[3].legend(handles=handles, labels=labels, loc='lower center',
              bbox_to_anchor=(-1.5, -0.45), fancybox=False, shadow=False,
              ncol=4, fontsize='xx-large')

plt.subplots_adjust(top=.9, bottom=0.3, right=0.99, left=0.05,
                    hspace=0, wspace=0.32)
plt.savefig('submission/info_in_D_rfgan_vs_baselines_K=10.pdf')
