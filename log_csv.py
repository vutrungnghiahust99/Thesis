
import os
import pandas as pd

root = "experiments"

header = [
    'lossg_mean', 'lossg_std', 'lossd_mean', 'lossd_std', 'dx_mean', 'dx_std', 'dgz_mean', 'dgz_std',
    'fid_score'
]

INPUT = [
    'gan1_10_heads_1G_10heads_diff_data_for_heads_sep_d'
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
