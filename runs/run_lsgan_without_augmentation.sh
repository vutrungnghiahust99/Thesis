python -m src.train_v1 --exp_name 'lsgan_baseline' \
                       --loss_name 'lsgan' \
                       --augmentation '0' \
                       --aug_times '-1' \
                       --dist 'gauss' \
                       --n_epochs '500' \
                       --n_heads '1' \
                       --interval '8'

python -m src.train_v1 --exp_name 'lsgan_10_heads_1G_10heads_DV2' \
                       --loss_name 'lsgan' \
                       --augmentation '0' \
                       --aug_times '-1' \
                       --dist 'gauss' \
                       --n_epochs '500' \
                       --n_heads '10' \
                       --interval '8'  \
                       --use_d_v2 '1'

python -m src.train_v1 --exp_name 'lsgan_10_heads_1G_10heads_diff_data_for_heads_DV2' \
                       --loss_name 'lsgan' \
                       --augmentation '0' \
                       --aug_times '-1' \
                       --dist 'gauss' \
                       --n_epochs '500' \
                       --n_heads '10' \
                       --interval '8' \
                       --diff_data_for_heads '1' \
                       --use_d_v2 '1'
