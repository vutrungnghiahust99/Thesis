import matplotlib.pyplot as plt
import os
import pandas as pd

root1 = 'experiments/rfgan'
exp1 = 'rfgan_lossgan1_bagging0_augmentation0_10heads_usemaskD'
exp2 = 'rfgan_lossgan1_bagging1_augmentation0_10heads_usemaskD'
exp3 = 'experiments/baseline/baseline_1head/baseline1head_lossgan1_augmentation0_usebighead0/baseline1head_lossgan1_augmentation0_usebighead0.csv'
exp4 = 'experiments/baseline/baseline_1head/baseline1head_lossgan1_augmentation0_usebighead1/baseline1head_lossgan1_augmentation0_usebighead1.csv'
exp5 = 'experiments/baseline/baseline_hgan/baselinehgan_losshgan_augmentation0/baselinehgan_losshgan_augmentation0.csv'

exp6 = 'experiments/rfgan/rfgan_lossgan1_bagging0_augmentation0_5heads_usemaskD_sagan/rfgan_lossgan1_bagging0_augmentation0_5heads_usemaskD_sagan.csv'
exp7 = 'experiments/rfgan/rfgan_lossgan1_bagging1_augmentation0_5heads_usemaskD_sagan/rfgan_lossgan1_bagging1_augmentation0_5heads_usemaskD_sagan.csv'
exp8 = 'experiments/baseline/baseline_hgan/baselinehgan_losshgan_5heads_augmentation0_sagan/baselinehgan_losshgan_5heads_augmentation0_sagan.csv'


df1 = pd.read_csv(os.path.join(root1, exp1, exp1 + '.csv'))
df2 = pd.read_csv(os.path.join(root1, exp2, exp2 + '.csv'))
df3 = pd.read_csv(exp3)
df4 = pd.read_csv(exp4)
df5 = pd.read_csv(exp5)

df6 = pd.read_csv(exp6)
df7 = pd.read_csv(exp7)
df8 = pd.read_csv(exp8)

n1 = min(df1.shape[0], df2.shape[0], df4.shape[0], df5.shape[0])
n2 = min(df6.shape[0], df7.shape[0], df8.shape[0])
print(n1)
print(n2)
df1 = df1.iloc[0:n1]
df2 = df2.iloc[0:n1]
df3 = df3.iloc[0:n1]
df4 = df4.iloc[0:n1]
df5 = df5.iloc[0:n1]

df6 = df6.iloc[0:n2]
df7 = df7.iloc[0:n2]
df8 = df8.iloc[0:n2]

print(df6['fid_score'].min())
print(df7['fid_score'].min())
print(df8['fid_score'].min())

epochs1 = [x * 4 for x in range(n1)]
epochs2 = [x * 4 for x in range(n2)]

colors = ['green', 'red', 'orange', 'blue', 'black']

fig, axs = plt.subplots(1, 4, sharex=False, sharey=False, figsize=(16, 5))

axs[0].plot(epochs1, df1['dx_mean'], color=colors[0], label='RFGan + no bagging')
axs[0].plot(epochs1, df2['dx_mean'], color=colors[1], label='RFGan + bagging')
axs[0].plot(epochs1, df3['dx_mean'], color=colors[2], label='Vanilla Gan')
axs[0].plot(epochs1, df4['dx_mean'], color=colors[3], label='Vanilla Gan + Big D')
axs[0].plot(epochs1, df5['dx_mean'], color=colors[4], label='HGan')
axs[0].set_xlabel('Epoch', fontsize='xx-large')
axs[0].set_ylabel(r'$F_d(x)$', fontsize='xx-large')
axs[0].set_title('Eriklindernoren', fontsize='xx-large')
axs[0].set_xticks([0, 200, 400])
axs[0].set_yticks([0.5, 0.7, 0.9])


axs[1].plot(epochs1, df1['dgz_mean'], color=colors[0])
axs[1].plot(epochs1, df2['dgz_mean'], color=colors[1])
axs[1].plot(epochs1, df3['dgz_mean'], color=colors[2])
axs[1].plot(epochs1, df4['dgz_mean'], color=colors[3])
axs[1].plot(epochs1, df5['dgz_mean'], color=colors[4])
axs[1].set_xlabel('Epoch', fontsize='xx-large')
axs[1].set_ylabel(r'$F_d(G(z))$', fontsize='xx-large')
axs[1].set_title('Eriklindernoren', fontsize='xx-large')
axs[1].set_xticks([0, 200, 400])
axs[1].set_yticks([0.1, 0.3, 0.5])

axs[2].plot(epochs2, df6['dx_mean'], color=colors[0])
axs[2].plot(epochs2, df7['dx_mean'], color=colors[1])
axs[2].plot(epochs2, df8['dx_mean'], color=colors[4])
axs[2].set_xlabel('Epoch', fontsize='xx-large')
axs[2].set_ylabel(r'$F_d(x)$', fontsize='xx-large')
axs[2].set_title('Sagan', fontsize='xx-large')
axs[2].set_xticks([0, 40, 80, 120])
axs[2].set_yticks([0.45, 0.55, 0.65])

axs[3].plot(epochs2, df6['dgz_mean'], color=colors[0])
axs[3].plot(epochs2, df7['dgz_mean'], color=colors[1])
axs[3].plot(epochs2, df8['dgz_mean'], color=colors[4])
axs[3].set_xlabel('Epoch', fontsize='xx-large')
axs[3].set_ylabel(r'$F_d(G(z))$', fontsize='xx-large')
axs[3].set_title('Sagan', fontsize='xx-large')
axs[3].set_xticks([0, 40, 80, 120])
axs[3].set_yticks([0.32, 0.41, 0.5])

handles, labels = axs[0].get_legend_handles_labels()
axs[3].legend(handles=handles, labels=labels, loc='lower center',
              bbox_to_anchor=(-1.5, -0.45), fancybox=False, shadow=False,
              ncol=5, fontsize='xx-large')

plt.subplots_adjust(top=.9, bottom=0.3, right=0.99, left=0.05,
                    hspace=0, wspace=0.32)
plt.savefig('submission/stablizing_augmentation_0_eriklindernoren_sagan.pdf')
