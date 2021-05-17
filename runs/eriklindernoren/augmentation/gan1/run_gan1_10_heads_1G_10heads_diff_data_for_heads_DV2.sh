python -m src.train_v1 --exp_name 'gan1_10_heads_1G_10heads_diff_data_for_heads_DV2' \
                       --loss_name 'gan1' \
                       --augmentation '1' \
                       --aug_times '5' \
                       --dist 'gauss' \
                       --n_epochs '1000' \
                       --n_heads '10' \
                       --interval '8' \
                       --diff_data_for_heads '1' \
                       --use_d_v2 '1'
