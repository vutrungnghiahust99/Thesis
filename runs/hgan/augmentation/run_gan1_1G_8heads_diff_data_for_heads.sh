python -m src.train_hgan --exp_name 'gan1_1G_8heads_diff_data_for_heads' \
                          --loss_name 'gan1' \
                          --augmentation '1' \
                          --aug_times '2' \
                          --dist 'gauss' \
                          --n_epochs '1000' \
                          --n_heads '8' \
                          --interval '2' \
                          --diff_data_for_heads '1'
