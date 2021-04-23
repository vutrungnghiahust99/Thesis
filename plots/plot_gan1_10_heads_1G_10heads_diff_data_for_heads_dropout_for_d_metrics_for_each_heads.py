import matplotlib.pyplot as plt
import pandas as pd

exp1 = 'experiments/gan1_10_heads_1G_10heads_diff_data_for_heads_dropout_for_d/gan1_10_heads_1G_10heads_diff_data_for_heads_dropout_for_d_metrics_for_each_heads.csv'
exp2 = 'experiments/gan1_10_heads_1G_10heads_diff_data_for_heads_dropout_for_d/gan1_10_heads_1G_10heads_diff_data_for_heads_dropout_for_d.csv'

df1 = pd.read_csv(exp1)
df2 = pd.read_csv(exp2)

indices = [x * 4 for x in range(df1.shape[0])]
df2 = df2.iloc[indices]
n = df1.shape[0]
epochs = [x * 8 for x in range(n)]

fig, axs = plt.subplots(3, 4, sharex=False, sharey=False, figsize=(16, 12))

for head_id in range(10):
    r = head_id // 4
    c = head_id % 4
    axs[r, c].plot(
        epochs,
        df1[f'dx_mean_{head_id}'],
        color='#000000',
        label='D(x)')
    axs[r, c].fill_between(
        epochs,
        df1[f'dx_mean_{head_id}'] - df1[f'dx_std_{head_id}'],
        df1[f'dx_mean_{head_id}'] + df1[f'dx_std_{head_id}'],
        color='#000000', alpha=0.2)

    axs[r, c].plot(
        epochs,
        df1[f'dgz_mean_{head_id}'],
        color='red',
        label='D(G(z))')
    axs[r, c].fill_between(
        epochs,
        df1[f'dgz_mean_{head_id}'] - df1[f'dgz_std_{head_id}'],
        df1[f'dgz_mean_{head_id}'] + df1[f'dgz_std_{head_id}'],
        color='red', alpha=0.2)

head_id = 0
axs[2, 3].plot(
    epochs,
    df2['dx_mean'],
    color='#000000',
    label='D(x)')
axs[2, 3].fill_between(
    epochs,
    df2['dx_mean'] - df2['dx_std'],
    df2['dx_mean'] + df2['dx_std'],
    color='#000000', alpha=0.2)
axs[2, 3].plot(
    epochs,
    df2['dgz_mean'],
    color='red',
    label='D(G(z))')
axs[2, 3].fill_between(
    epochs,
    df2['dgz_mean'] - df2['dgz_std'],
    df2['dgz_mean'] + df2['dgz_std'],
    color='red', alpha=0.2)


handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', fontsize='large')
plt.tight_layout()
plt.savefig('reports/gan1_10_heads_1G_10heads_diff_data_for_heads_dropout_for_d_metrics_for_each_heads.pdf')
