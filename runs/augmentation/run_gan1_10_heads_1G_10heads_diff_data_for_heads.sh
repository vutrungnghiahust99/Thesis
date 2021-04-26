python -m src.train_v1 --exp_name 'gan1_10_heads_1G_10heads_diff_data_for_heads' \
                       --loss_name 'gan1' \
                       --augmentation '1' \
                       --aug_times '5' \
                       --dist 'gauss' \
                       --n_epochs '300' \
                       --n_heads '10' \
                       --interval '4' \
                       --diff_data_for_heads '1'
