
import os
import pandas as pd

root = "experiments/augmentation"

header = [
    'lossg_mean', 'lossg_std', 'lossd_mean', 'lossd_std', 'dx_mean', 'dx_std', 'dgz_mean', 'dgz_std',
    'fid_score',
    'lossg_mean_0', 'lossg_std_0', 'lossd_mean_0', 'lossd_std_0', 'dx_mean_0', 'dx_std_0', 'dgz_mean_0', 'dgz_std_0',
    'lossg_mean_1', 'lossg_std_1', 'lossd_mean_1', 'lossd_std_1', 'dx_mean_1', 'dx_std_1', 'dgz_mean_1', 'dgz_std_1',
    'lossg_mean_2', 'lossg_std_2', 'lossd_mean_2', 'lossd_std_2', 'dx_mean_2', 'dx_std_2', 'dgz_mean_2', 'dgz_std_2',
    'lossg_mean_3', 'lossg_std_3', 'lossd_mean_3', 'lossd_std_3', 'dx_mean_3', 'dx_std_3', 'dgz_mean_3', 'dgz_std_3',
    'lossg_mean_4', 'lossg_std_4', 'lossd_mean_4', 'lossd_std_4', 'dx_mean_4', 'dx_std_4', 'dgz_mean_4', 'dgz_std_4',
    'lossg_mean_5', 'lossg_std_5', 'lossd_mean_5', 'lossd_std_5', 'dx_mean_5', 'dx_std_5', 'dgz_mean_5', 'dgz_std_5',
    'lossg_mean_6', 'lossg_std_6', 'lossd_mean_6', 'lossd_std_6', 'dx_mean_6', 'dx_std_6', 'dgz_mean_6', 'dgz_std_6',
    'lossg_mean_7', 'lossg_std_7', 'lossd_mean_7', 'lossd_std_7', 'dx_mean_7', 'dx_std_7', 'dgz_mean_7', 'dgz_std_7',
    'lossg_mean_8', 'lossg_std_8', 'lossd_mean_8', 'lossd_std_8', 'dx_mean_8', 'dx_std_8', 'dgz_mean_8', 'dgz_std_8',
    'lossg_mean_9', 'lossg_std_9', 'lossd_mean_9', 'lossd_std_9', 'dx_mean_9', 'dx_std_9', 'dgz_mean_9', 'dgz_std_9'
]

INPUT = [
    'gan1_10_heads_1G_10heads'
]


def get_numeric(s):
    s = s.split('\n')
    s = [x for x in s if '[' in x and ']' in x and 'accuracy' not in x and '/s' not in x and '_' not in x]
    result = []
    for line in s:
        line = line.split('[')[1]
        line = line.split(']')[0]
        line = line.split(',')
        line = [float(x) for x in line]
        result = result + line
    assert len(result) == len(header) or ((len(result) - len(header)) % 8) == 0
    return result[0:len(header)]

for inp in INPUT:
    int_file = os.path.join(root, inp, inp + '.txt')
    OUTPUT = os.path.join(root, inp, inp + '.csv')

    logs = open(int_file).read()
    logs = logs.split('--------------------------------')
    logs = logs[1:-1]
    results = [get_numeric(x) for x in logs]
    df = pd.DataFrame(results, columns=header)
    print(df.shape)
    df.to_csv(OUTPUT, index=None)
    print(OUTPUT)
    print('Completed')
