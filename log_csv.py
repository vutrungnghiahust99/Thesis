
import os
import pandas as pd

root = "experiments/rfgan/augmentation"

# header = [
#     'lossg_mean', 'lossg_std', 'lossd_mean', 'lossd_std', 'dx_mean', 'dx_std', 'dgz_mean', 'dgz_std',
#     'fid_score'
# ]

header = [
    'lossg_mean', 'lossg_std', 'lossd_mean', 'lossd_std',
    'dx_mean', 'dx_std', 'dx_min', 'dx_max', 'dx_max_min',
    'dgz_mean', 'dgz_std', 'dgz_min', 'dgz_max', 'dgz_max_min',
    'fid_score',
]

INPUT = [
    'rfgan_lossgan1_bagging0_augmentation1_10heads_usemaskD',
    'rfgan_lossgan1_bagging1_augmentation1_10heads_usemaskD',
]


def get_numeric(s):
    s = s.split('\n')
    s = [x for x in s if '[' in x and ']' in x and 'accuracy' not in x and '/s' not in x and '_' not in x and (len(x) < 700)]
    result = []
    for line in s:
        line = line.split('[')[1]
        line = line.split(']')[0]
        line = line.split(',')
        line = [float(x) for x in line]
        result = result + line
    assert len(result) == len(header), f'{len(result)} != {len(header)}'
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
