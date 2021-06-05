import matplotlib.pyplot as plt
import os
import pandas as pd

root1 = 'experiments/rfgan'
exp1 = 'rfgan_lossgan1_bagging0_augmentation0_3heads_usemaskD'
exp2 = 'rfgan_lossgan1_bagging0_augmentation0_5heads_usemaskD'
exp3 = 'rfgan_lossgan1_bagging0_augmentation0_10heads_usemaskD'

df1 = pd.read_csv(os.path.join(root1, exp1, exp1 + '.csv'))
df2 = pd.read_csv(os.path.join(root1, exp2, exp2 + '.csv'))
df3 = pd.read_csv(os.path.join(root1, exp3, exp3 + '.csv'))

n = min(df1.shape[0], df2.shape[0], df3.shape[0])
print(n)
df1 = df1.iloc[0:n]
df2 = df2.iloc[0:n]
df3 = df3.iloc[0:n]

print(df1['fid_score'].min())
print(df2['fid_score'].min())
print(df3['fid_score'].min())
epochs = [x * 4 for x in range(n)]

fig, axs = plt.subplots(4, 3, sharex=False, sharey=False, figsize=(12, 14))

df = [df1, df2, df3]
colors = ['green', 'blue', 'red']
labels = ['K=3', 'K=5', 'K=10']

# marker symbol
n1 = 40
for i in range(3):
    axs[0, 0].plot(epochs[n1:], df[i]['fid_score'][n1:], color=colors[i])
axs[0, 0].set_xlabel('epoch', fontsize='x-large')
axs[0, 0].set_ylabel('FID', fontsize='xx-large')

for i in range(3):
    axs[0, 1].plot(epochs, df[i]['fid_score'], color=colors[i])
axs[0, 1].set_xlabel('epoch', fontsize='x-large')
axs[0, 1].set_ylabel('FID', fontsize='xx-large')

n2 = 0
for i in range(3):
    axs[0, 2].plot(epochs[n2:], df[i][n2:]['dx_mean'], color=colors[i])
axs[0, 2].set_xlabel('epoch', fontsize='x-large')
axs[0, 2].set_ylabel(r'$D(x)$', fontsize='xx-large')

for i in range(3):
    axs[1, 0].plot(epochs[n2:], df[i][n2:]['dgz_mean'], color=colors[i], label=labels[i])
axs[1, 0].set_xlabel('epoch', fontsize='x-large')
axs[1, 0].set_ylabel(r'$D(G(z))$', fontsize='large')

for i in range(3):
    axs[1, 1].plot(epochs, df[i]['lossg_mean'], color=colors[i])
axs[1, 1].set_xlabel('epoch', fontsize='x-large')
axs[1, 1].set_ylabel(r'$V(G)$', fontsize='large')

for i in range(3):
    axs[1, 2].plot(epochs, df[i]['lossd_mean'], color=colors[i])
axs[1, 2].set_xlabel('epoch', fontsize='x-large')
axs[1, 2].set_ylabel(r'$V(D)$', fontsize='large')

for i in range(3):
    axs[2, 0].plot(epochs, df[i]['dx_std'], color=colors[i])
axs[2, 0].set_xlabel('epoch', fontsize='x-large')
axs[2, 0].set_ylabel(r'$D(x)_{std}$', fontsize='large')

for i in range(3):
    axs[2, 1].plot(epochs, df[i]['dgz_std'], color=colors[i])
axs[2, 1].set_xlabel('epoch', fontsize='x-large')
axs[2, 1].set_ylabel(r'$D(G(z))_{std}$', fontsize='large')

for i in range(3):
    axs[2, 2].plot(epochs, df[i]['dx_max_min'], color=colors[i])
axs[2, 2].set_xlabel('epoch', fontsize='x-large')
axs[2, 2].set_ylabel(r'$D(x)_{max} - D(x)_{min}$', fontsize='large')

for i in range(3):
    axs[3, 0].plot(epochs, df[i]['dgz_max_min'], color=colors[i])
axs[3, 0].set_xlabel('epoch', fontsize='x-large')
axs[3, 0].set_ylabel(r'$D(G(z))_{max} - D(G(z))_{min}$', fontsize='large')

handles, labels = axs[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', fontsize='large')
plt.tight_layout()
plt.savefig('plots/rfgan/bagging0_augmentation0_K_3_5_10.pdf')
