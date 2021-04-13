import matplotlib.pyplot as plt
import pandas as pd

exp1 = 'experiments/gan1_10_heads_1G_10heads/gan1_10_heads_1G_10heads_metrics_for_each_heads.csv'

df1 = pd.read_csv(exp1)
n = df1.shape[0]
epochs = [x * 8 for x in range(n)]

fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(12, 12))

for head_id in range(9):
    r = head_id // 3
    c = head_id % 3
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

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', fontsize='large')
plt.tight_layout()
plt.savefig('reports/gan1_10_heads_1G_10heads_metrics_for_each_heads.pdf')
