import matplotlib.pyplot as plt
import pandas as pd

exp1 = 'experiments/gan1_10_heads_1G_10heads_diff_data_for_heads/gan1_10_heads_1G_10heads_diff_data_for_heads.csv'
exp2 = 'experiments/gan1_10_heads_1G_10heads_diff_data_for_heads_sep_d/gan1_10_heads_1G_10heads_diff_data_for_heads_sep_d.csv'

df1 = pd.read_csv(exp1)
df2 = pd.read_csv(exp2)
n = min(df1.shape[0], df2.shape[0])
df1 = df1.iloc[0:n, :]
df2 = df2.iloc[0:n, :]
epochs = [x * 2 for x in range(n)]

fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(12, 8))

n = 50
df1_ = df1.iloc[n:, :]
df2_ = df2.iloc[n:, :]
epochs_ = epochs[n:]
axs[0, 0].plot(epochs_, df1_['fid_score'], 'k--', color='green', label='10 heads + diff_data')
axs[0, 0].plot(epochs_, df2_['fid_score'], 'k:', color='blue', label='10 heads + diff_data + sep_d')
axs[0, 0].set_xlabel('epoch', fontsize='x-large')
axs[0, 0].set_ylabel('FID', fontsize='large')

axs[0, 1].plot(epochs, df1['fid_score'], 'k--', color='green')
axs[0, 1].plot(epochs, df2['fid_score'], 'k:', color='blue')
axs[0, 1].set_xlabel('epoch', fontsize='x-large')
axs[0, 1].set_ylabel('FID', fontsize='large')

axs[0, 2].plot(epochs, df1['lossd_mean'], 'k--', color='green')
axs[0, 2].plot(epochs, df2['lossd_mean'], 'k:', color='blue')
axs[0, 2].set_xlabel('epoch', fontsize='x-large')
axs[0, 2].set_ylabel(r'$V(D)$', fontsize='large')

axs[1, 0].plot(epochs, df1['dgz_mean'] * -1, 'k--', color='green')
axs[1, 0].plot(epochs, df2['dgz_mean'] * -1, 'k:', color='blue')
axs[1, 0].set_xlabel('epoch', fontsize='x-large')
axs[1, 0].set_ylabel(r'$-D(G(z))$', fontsize='large')

axs[1, 1].plot(epochs, df1['dx_mean'], 'k--', color='green')
axs[1, 1].plot(epochs, df2['dx_mean'], 'k:', color='blue')
axs[1, 1].set_xlabel('epoch', fontsize='x-large')
axs[1, 1].set_ylabel(r'$D(x)$', fontsize='large')

axs[1, 2].plot(epochs, df1['lossg_mean'], 'k--', color='green')
axs[1, 2].plot(epochs, df2['lossg_mean'], 'k:', color='blue')
axs[1, 2].set_xlabel('epoch', fontsize='x-large')
axs[1, 2].set_ylabel(r'$V(G)$', fontsize='large')


handles, labels = axs[0, 0].get_legend_handles_labels()
axs[1, 1].legend(handles=handles, labels=labels, loc='lower center',
                 bbox_to_anchor=(0.45, -0.5), fancybox=False, shadow=False,
                 ncol=4, fontsize='xx-large')
plt.subplots_adjust(bottom=0.2, wspace=0.32, top=0.95, left=0.07, right=0.98, hspace=0.25)

plt.savefig('reports/gan1_10_heads_1G_10heads_diff_data_for_heads__gan1_10_heads_1G_10heads_diff_data_for_heads_sep_d.pdf')
