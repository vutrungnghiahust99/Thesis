python -m src.train_sep_d --exp_name 'gan1_10_heads_1G_10heads_diff_data_for_heads_sep_d' \
                          --loss_name 'gan1' \
                          --augmentation '0' \
                          --spec_g '0' \
                          --spec_d '1' \
                          --dist 'gauss' \
                          --n_epochs '300' \
                          --n_heads '10' \
                          --interval '2' \
                          --diff_data_for_heads '1'
