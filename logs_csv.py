import os
import pandas as pd

root = "experiments/eriklindernoren_architecture/augmentation_v2/add_gaussian_noise/wgangp"

header = [
    'g_wrt_z_mean', 'g_wrt_z_std', 'g_wrt_z_min', 'g_wrt_z_max',
    'lossg_wrt_theta_g_mean', 'lossg_wrt_theta_g_std', 'lossg_wrt_theta_g_min', 'lossg_wrt_theta_g_max',
    'lossd_wrt_theta_d_mean', 'lossd_wrt_theta_d_std', 'lossd_wrt_theta_d_min', 'lossd_wrt_theta_d_max',
    'd_wrt_gz_mean', 'd_wrt_gz_std', 'd_wrt_gz_min', 'd_wrt_gz_max',
    'lossg_wrt_z_mean', 'lossg_wrt_z_std', 'lossg_wrt_z_min', 'lossg_wrt_zmax',
    'lossg_mean', 'lossg_std', 'lossd_mean', 'lossd_std', 'dx_mean', 'dx_std', 'dgz_mean', 'dgz_std',
    'fid_score',
]

INPUT = [
    'exp_id_augmentation_v2_gaussian_noise_base_exp_name_eriklindernoren_architecture_loss_name_wgangp_shift_-1_spec_g_0_spec_d_0_dist_gauss_std_-1.0_bound_1.0_aug_1_aug_times_0_train_g_more_0_bs_g_-1_n_epochs_501_n_critic_5_interval_10',
    'exp_id_augmentation_v2_gaussian_noise_base_exp_name_eriklindernoren_architecture_loss_name_wgangp_shift_-1_spec_g_0_spec_d_0_dist_gauss_std_0.5_bound_1.0_aug_1_aug_times_128_train_g_more_0_bs_g_-1_n_epochs_501_n_critic_5_interval_10',
    'exp_id_augmentation_v2_gaussian_noise_base_exp_name_eriklindernoren_architecture_loss_name_wgangp_shift_-1_spec_g_0_spec_d_0_dist_gauss_std_0.1_bound_1.0_aug_1_aug_times_64_train_g_more_0_bs_g_-1_n_epochs_501_n_critic_5_interval_10',
    'exp_id_augmentation_v2_gaussian_noise_base_exp_name_eriklindernoren_architecture_loss_name_wgangp_shift_-1_spec_g_0_spec_d_0_dist_gauss_std_0.01_bound_1.0_aug_1_aug_times_16_train_g_more_0_bs_g_-1_n_epochs_501_n_critic_5_interval_10',
    'exp_id_augmentation_v2_gaussian_noise_base_exp_name_eriklindernoren_architecture_loss_name_wgangp_shift_-1_spec_g_0_spec_d_0_dist_gauss_std_0.001_bound_1.0_aug_1_aug_times_4_train_g_more_0_bs_g_-1_n_epochs_501_n_critic_5_interval_10'
]


def get_numeric(s):
    s = s.split('\n')
    s = [x for x in s if '[' in x and ']' in x and 'accuracy' not in x]
    result = []
    for line in s:
        line = line.split('[')[1]
        line = line.split(']')[0]
        line = line.split(',')
        line = [float(x) for x in line]
        result = result + line
    assert len(result) == len(header)
    return result

for inp in INPUT:
    int_file = os.path.join(root, inp, inp + '.txt')
    OUTPUT = os.path.join(root, inp, inp + '.csv')

    logs = open(int_file).read()
    logs = logs.split('--------------------------------')
    logs = logs[1:-1]
    results = [get_numeric(x) for x in logs]
    df = pd.DataFrame(results, columns=header)
    # if os.path.isfile(OUTPUT):
    #     raise RuntimeError(f'{OUTPUT} exists')
    df.to_csv(OUTPUT, index=None)
    print(OUTPUT)
    print('Completed')
