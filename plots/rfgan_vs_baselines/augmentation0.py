import matplotlib.pyplot as plt
import os
import pandas as pd

root1 = 'experiments/rfgan'
exp1 = 'rfgan_lossgan1_bagging0_augmentation0_10heads_usemaskD'
exp2 = 'rfgan_lossgan1_bagging1_augmentation0_10heads_usemaskD'
exp3 = 'experiments/baseline/baseline_1head/baseline1head_lossgan1_augmentation0_usebighead0/baseline1head_lossgan1_augmentation0_usebighead0.csv'
exp4 = 'experiments/baseline/baseline_1head/baseline1head_lossgan1_augmentation0_usebighead1/baseline1head_lossgan1_augmentation0_usebighead1.csv'
exp5 = 'experiments/baseline/baseline_mheads/baselinemheads_lossgan1_augmentation0/baselinemheads_lossgan1_augmentation0.csv'
exp6 = 'experiments/baseline/baseline_hgan/baselinehgan_losshgan_augmentation0/baselinehgan_losshgan_augmentation0.csv'

df1 = pd.read_csv(os.path.join(root1, exp1, exp1 + '.csv'))
df2 = pd.read_csv(os.path.join(root1, exp2, exp2 + '.csv'))
df3 = pd.read_csv(exp3)
df4 = pd.read_csv(exp4)
df5 = pd.read_csv(exp5)
df6 = pd.read_csv(exp6)

n = min(df1.shape[0], df2.shape[0], df3.shape[0], df4.shape[0], df5.shape[0], df6.shape[0])
df1 = df1.iloc[0:n]
df2 = df2.iloc[0:n]
df3 = df3.iloc[0:n]
df4 = df4.iloc[0:n]
df5 = df5.iloc[0:n]
df6 = df6.iloc[0:n]

print(n)
print(df1['fid_score'].min())
print(df2['fid_score'].min())
print(df3['fid_score'].min())
print(df4['fid_score'].min())
print(df5['fid_score'].min())
print(df6['fid_score'].min())

epochs = [x * 4 for x in range(n)]

fig, axs = plt.subplots(4, 3, sharex=False, sharey=False, figsize=(12, 14))

df = [df1, df2, df3, df4, df5, df6]
colors = ['green', 'blue', 'red', 'orange', 'black', 'purple']
labels = ['rfgan', 'rfgan+bagging', '1smallhead', '1bighead', 'multipleDs', 'hgan']

# marker symbol
n1 = 70
for i in range(6):
    axs[0, 0].plot(epochs[n1:], df[i]['fid_score'][n1:], color=colors[i])
axs[0, 0].set_xlabel('epoch', fontsize='x-large')
axs[0, 0].set_ylabel('FID', fontsize='xx-large')

for i in range(6):
    axs[0, 1].plot(epochs, df[i]['fid_score'], color=colors[i])
axs[0, 1].set_xlabel('epoch', fontsize='x-large')
axs[0, 1].set_ylabel('FID', fontsize='xx-large')

n2 = 0
for i in range(6):
    axs[0, 2].plot(epochs[n2:], df[i][n2:]['dx_mean'], color=colors[i])
axs[0, 2].set_xlabel('epoch', fontsize='x-large')
axs[0, 2].set_ylabel(r'$D(x)$', fontsize='xx-large')

for i in range(6):
    axs[1, 0].plot(epochs[n2:], df[i][n2:]['dgz_mean'], color=colors[i], label=labels[i])
axs[1, 0].set_xlabel('epoch', fontsize='x-large')
axs[1, 0].set_ylabel(r'$D(G(z))$', fontsize='large')

for i in range(6):
    axs[1, 1].plot(epochs, df[i]['lossg_mean'], color=colors[i])
axs[1, 1].set_xlabel('epoch', fontsize='x-large')
axs[1, 1].set_ylabel(r'$V(G)$', fontsize='large')

for i in range(6):
    axs[1, 2].plot(epochs, df[i]['lossd_mean'], color=colors[i])
axs[1, 2].set_xlabel('epoch', fontsize='x-large')
axs[1, 2].set_ylabel(r'$V(D)$', fontsize='large')

for i in [0, 1, 4, 5]:
    axs[2, 0].plot(epochs, df[i]['dx_std'], color=colors[i])
axs[2, 0].set_xlabel('epoch', fontsize='x-large')
axs[2, 0].set_ylabel(r'$D(x)_{std}$', fontsize='large')

for i in [0, 1, 4, 5]:
    axs[2, 1].plot(epochs, df[i]['dgz_std'], color=colors[i])
axs[2, 1].set_xlabel('epoch', fontsize='x-large')
axs[2, 1].set_ylabel(r'$D(G(z))_{std}$', fontsize='large')

for i in [0, 1, 4, 5]:
    axs[2, 2].plot(epochs, df[i]['dx_max_min'], color=colors[i])
axs[2, 2].set_xlabel('epoch', fontsize='x-large')
axs[2, 2].set_ylabel(r'$D(x)_{max} - D(x)_{min}$', fontsize='large')

for i in [0, 1, 4, 5]:
    axs[3, 0].plot(epochs, df[i]['dgz_max_min'], color=colors[i])
axs[3, 0].set_xlabel('epoch', fontsize='x-large')
axs[3, 0].set_ylabel(r'$D(G(z))_{max} - D(G(z))_{min}$', fontsize='large')

handles, labels = axs[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', fontsize='large')
plt.tight_layout()
plt.savefig('plots/rfgan_vs_baselines/augmentation0.pdf')
